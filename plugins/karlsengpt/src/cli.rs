//use crate::Error;
//use std::str::FromStr;



#[derive(clap::Args, Debug, Clone)]
pub struct KarlsenGptOpt {
    #[clap(long = "karlsengpt-disable", help = "Disable Karlsen GPT")]
    pub karlsengpt_disable: bool,
    #[clap(long = "karlsengpt-model-path", help = "path to the model file (.gguf)", default_value = ".")]
    pub karlsengpt_model_path: String,
}
