use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn whos_that_dog() -> String {
    "Mister Peanutbutter".into()
}

#[cfg(test)]
mod tests {
    use super::*;
}
