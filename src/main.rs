use safetensors::SafeTensors;
use tokenizers::Tokenizer;

fn main() {
    let buffer = std::fs::read("model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors").expect("Failed to read file");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    println!("Tensors: {:?}", tensors.names());
    let tokenizer = Tokenizer::from_file("model/Qwen3.5-0.8B/tokenizer.json").expect("Failed to load tokenizer");
    let encoded = tokenizer.encode("你好，Hello World", false).expect("Failed to encode input");
    println!("Encoded IDs: {:?}", encoded.get_ids());
}
