
How to run (example):

```
poetry shell

PYTHONPATH=$PWD time manim -p watersort/main.py S10MutatingPuzzle -ql
```

How to run the rust solver (release build):

```
cargo build --release && time ./target/release/watersort_rust
```
