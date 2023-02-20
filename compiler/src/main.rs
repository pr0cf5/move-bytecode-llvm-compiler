use clap::Parser;
use compiler::TranslationUnit;
use inkwell::context::Context;
use move_binary_format::file_format::CompiledModule;
use std::{
    fs::File,
    io::{Read, Write},
};

#[derive(Parser)]
struct Args {
    #[arg(long, short)]
    input_file: String,
    #[arg(long, short)]
    output_file: String,
}

mod compiler;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let context = Context::create();
    let mut input_file = File::open(&args.input_file)?;
    let mut input_bytes = vec![];
    input_file.read_to_end(&mut input_bytes)?;
    let module = CompiledModule::deserialize(&input_bytes)?;
    let tr = TranslationUnit::new(&context, "module");
    let bytes = tr.translate(&module)?;
    let mut file = File::create(&args.output_file)?;
    file.write_all(&bytes)?;
    Ok(())
}
