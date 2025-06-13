use std::{fs, num::NonZeroU32, path::PathBuf, pin::pin};

use crate::Error;
use anyhow::Context;
//use hf_hub::api::sync::ApiBuilder;
use karlsen_miner::Worker;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::data_array::LlamaTokenDataArray,
};
use log::info;

//static BPS: f32 = 0.5;

/*
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
        repo: String,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}
*/

pub struct KarlsenGptGPUWorker {
    // NOTE: The order is important! context must be closed last
    device_id: u32,
    model_path: String,
}

impl Worker for KarlsenGptGPUWorker {
    fn id(&self) -> String {
        format!("#{}", self.device_id)
    }

    fn compute_pouw(&self, task: &String) -> Result<String, Error> {
        info!("processing pouw task : {}", task);

        // init LLM
        let backend = LlamaBackend::init()?;
        // offload all layers to the gpu
        let model_params = {
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            if !disable_gpu {
                LlamaModelParams::default().with_n_gpu_layers(1000)
            } else {
                LlamaModelParams::default()
            }
            #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
            LlamaModelParams::default()
        };
        let prompt = task.clone();
        let mut model_params = pin!(model_params);
        /*
        for (k, v) in &key_value_overrides {
            let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }
        */

        info!("LLM engine batch loading model ...");
        //let model_path = PathBuf::from("D:\\work\\karlsen\\tensorflow\\llama-cpp-rs\\llama-cpp-rs\\llama-2-7b.Q4_K_M.gguf");

        // Get the current directory
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        // Find the first *.gguf file in the current directory
        let model_path = fs::read_dir(&current_dir)
            .expect("Failed to read directory")
            .filter_map(|entry| entry.ok()) // Filter out errors
            .map(|entry| entry.path()) // Get the paths
            .find(|path| path.extension().map_or(false, |ext| ext == "gguf")) // Find the first file with .gguf extension
            .expect("No .gguf file found in the current directory");
        let model =
            LlamaModel::load_from_file(&backend, model_path, &model_params).with_context(|| "unable to load model")?;

        // initialize the context
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
        //TODO: comment for debug
        //.with_seed(1234);
        /*
        if let Some(threads) = threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = threads_batch.or(threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }
        */

        info!("LLM engine context init ...");
        let mut ctx = model.new_context(&backend, ctx_params).with_context(|| "unable to create the llama_context")?;

        info!("LLM engine token generation ...");
        let tokens_list =
            model.str_to_token(&prompt, AddBos::Always).with_context(|| format!("failed to tokenize {prompt}"))?;

        let n_len = 32;
        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            return Err(
                "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_len or increase n_ctx"
                    .into(),
            );
        }

        if tokens_list.len() >= usize::try_from(n_len)? {
            return Err("the prompt is too long, it has more tokens than n_len".into());
        }

        info!("LLM engine batch init ...");
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(512, 1);

        info!("LLM engine batch adding tokens ...");
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        info!("LLM engine decoding ...");
        ctx.decode(&mut batch).with_context(|| "llama_decode() failed")?;

        info!("LLM engine decode done");
        // main loop

        let mut n_cur = batch.n_tokens();
        //let mut n_decode = 0;

        //let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut result = String::from("");

        info!("starting LLM engine");
        while n_cur <= n_len {
            // sample the next token
            {
                let candidates = ctx.candidates();

                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                //TODO: comment for
                /*
                // sample the most likely token
                let new_token_id = ctx.sample_token_greedy(candidates_p);

                // is it an end of stream?
                if model.is_eog_token(new_token_id) {
                    eprintln!();
                    break;
                }

                let output_bytes = model.token_to_bytes(new_token_id, llama_cpp_2::model::Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                //print!("{output_string}");
                //std::io::stdout().flush()?;
                result.push_str(&output_string);

                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true)?;
                */
            }
            n_cur += 1;

            ctx.decode(&mut batch).with_context(|| "failed to eval")?;

            //n_decode += 1;
        }
        info!("LLM engine done : {}", result);

        Ok(format!("{result}"))
        //format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, _nonce_mask: u64, _nonce_fixed: u64) {
        //println!("calculate_hash");
    }

    fn load_block_constants(&mut self, _hash_header: &[u8; 72], _target: &[u64; 4]) {
        //println!("load_block_constants");
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        //println!("sync");
        Ok(())
    }

    fn get_workload(&self) -> usize {
        //println!("get_workload");
        1
    }

    #[inline(always)]
    fn copy_output_to(&mut self, _nonces: &mut Vec<u64>) -> Result<(), Error> {
        //println!("copy_output_to");
        Ok(())
    }
}

impl<'gpu> KarlsenGptGPUWorker {
    pub fn new(device_id: u32, model_path: String) -> Result<Self, Error> {
        info!("Starting a GPT worker");
        Ok(Self { device_id, model_path })
    }
}
