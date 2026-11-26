# textkit

Small, dependency-free text utilities.

```python
>>> from textkit import slugify, title_case, truncate, render, top_words
>>> slugify("Crème Brûlée Recipe")
'creme-brulee-recipe'
>>> title_case("the lord of the rings")
'The Lord of the Rings'
>>> truncate("A fairly long sentence about nothing", 20)
'A fairly long sen...'
>>> render("Hello {user.name}!", user={"name": "Ada"})
'Hello Ada!'
>>> top_words("the cat and the hat", n=1)
[('the', 2)]
```

## Modules

| module              | what it does                                              |
|---------------------|-----------------------------------------------------------|
| `textkit.slug`      | `slugify()` for URLs and file names                       |
| `textkit.wrapping`  | `wrap()`, `fill()`, `indent()`                            |
| `textkit.tokenize`  | `word_tokens()`, `sentence_split()`, `ngrams()`           |
| `textkit.stats`     | word counts, frequencies, reading time                    |
| `textkit.case`      | `snake_case()`, `camel_case()`, `title_case()` and friends|
| `textkit.truncation`| `truncate()`, `truncate_words()`, `truncate_middle()`     |
| `textkit.csvio`     | parse and format single CSV lines, read small CSV files   |
| `textkit.template`  | `{name}` placeholder templates                            |
| `textkit.normalize` | whitespace, quote and accent normalisation                |
| `textkit.cli`       | the `textkit` command line tool                           |

## Command line

```
textkit slug "Hello World"
textkit wrap --width 40 < essay.txt
textkit stats essay.txt
textkit render "Dear {name}," name=Ada
```

## Development

```
pip install -e .[test]
pytest
```
