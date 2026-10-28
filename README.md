# Markov Me (static site)

A browser version of `../generate.py`. `build.py` trains the chain and exports it to `model.json`: which words follow each 1–2 word context, and how often. `markov.js` runs it in the browser with the same backoff, typo correction and quote check as the Python version. The corpus text itself isn't published.

## Update the corpus

```sh
python3 markov/web/build.py
```

Rerun it whenever the corpus, `FOLDER_WEIGHTS` or `QUOTE_STREAK` change, then commit `model.json`.

## Preview locally

```sh
python3 -m http.server -d markov/web 8000   # then open http://localhost:8000
```

Opening `index.html` directly from disk won't work, because browsers block `fetch` on `file://` pages.

## Deploy to GitHub Pages

This folder is its own git repo. For a user site, the GitHub repo must be named `<username>.github.io`:

```sh
cd markov/web
gh repo create wharrison2.github.io --public --source . --push
```

GitHub serves it at https://wharrison2.github.io a minute or so after the push.

Note: `model.json` doesn't contain the corpus files, but anyone determined could rebuild long stretches of them from it, because many 2-word contexts have only one possible next word.
