fn main() {
    println!("cargo:rustc-link-search=native=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=static=main5_multi");
    println!("cargo:rustc-link-lib=static=main6_multi");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
