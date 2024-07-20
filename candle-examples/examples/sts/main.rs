use candle_nn::VarBuilder;
use clap::Parser;
use candle_transformers::models::bert_sts::{BertSTSModel, Config, HiddenAct, DTYPE};
use std::{collections::HashMap, fs::read_to_string, path::Path};
use candle::Tensor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use anyhow::{Error as E, Result};
use tokenizers::{PaddingParams, Token, Tokenizer};
#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg[long]]
    cpu: bool,

    #[arg[long]]
    tracing: bool,

    #[arg[long]]
    model_id: Option<String>,

    #[arg[long]]
    revision: Option<String>,

    #[arg[long]]
    prompt1: Option<String>,
    
    #[arg[long]]
    prompt2: Option<String>,

    #[arg[long]]
    use_pth: bool,

    #[arg(long, default_value = "1")]
    n: usize,

    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    #[arg(long, default_value="false")]
    approximate_gelu: bool,
}

impl Args {
    fn build_model_and_tokenizer_config(&self) -> Result<(BertSTSModel, Tokenizer, Config)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "sentence-transformers/LaBSE".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),

            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision)
        };
        let model_path = Path::new(& model_id);
        let (config_filename, tokenizer_filename, weight_filename)  = if model_path.exists() {
            let config_filename = model_path.join("config.json");
            let tokenizer_filename = model_path.join("tokenizer.json");
            let weight_filename = if self.use_pth {
                model_path.join("pytorch_model.bin")
            } else {
                model_path.join("model.safetensors")
            };
            (config_filename, tokenizer_filename, weight_filename)
        } else {
            let repo = Repo::with_revision(model_id, RepoType::Model, revision);
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = if self.use_pth {
            VarBuilder::from_pth(&weight_filename, DTYPE, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weight_filename], DTYPE, &device)?
            }
        };

        let model = BertSTSModel::load(vb, &config)?;
        Ok((model, tokenizer, config))

    }
}


fn get_token_representation(tokenizer: &mut Tokenizer,
                             prompt: String, 
                             device: &candle::Device, 
                             start: std::time::Instant) -> Result<(Tensor, Tensor)>  {
    let tokenizer = tokenizer
                                .with_padding(None)
                                .with_truncation(None)
                                .map_err(E::msg)?;
    let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("Loaded and encoded {:?}", start.elapsed());

    Ok((token_ids, token_type_ids))
}

// fn get_cosine_similarity(ts1:Tensor, ts2: Tensor) {

// }


fn main() -> Result<()> {

    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();
    let (model, mut tokenizer, config) = args.build_model_and_tokenizer_config()?;
    let device = &model.device; 
    if let (Some(prompt1), Some(prompt2)) = (args.prompt1, args.prompt2) {

        let (prompt1_tokens, prompt1_token_ids) = get_token_representation(& mut tokenizer, prompt1, device, start).unwrap();
        let (prompt2_tokens, prompt2_token_ids) = get_token_representation(& mut tokenizer, prompt2, device, start).unwrap();
        let prompt1_token_type_ids = prompt1_token_ids.zeros_like()?;
        let prompt2_token_type_ids  =  prompt2_token_ids.zeros_like()?;

        let start = std::time::Instant::now();
        let ys1 = model.forward(&prompt1_token_ids, &prompt1_token_type_ids);
        let ys2 = model.forward(&prompt2_token_ids, &prompt2_token_type_ids);




    }
//    Result()
    Ok(())

}