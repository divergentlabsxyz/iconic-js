# Iconic for JavaScript

Pick a [Phosphor](https://phosphoricons.com) icon for a short note or to-do.
Runs locally in the browser and in Node, with no dependencies.

This package runs the [`amsintelligence/iconic`](https://huggingface.co/amsintelligence/iconic)
model. It works in the 24 official EU languages.

## Install

```bash
npm install github:amsintelligence/iconic-js
```

Node 18 or newer, or any modern browser.

## Use in Node

The model files ship inside the package, so nothing is downloaded.

```js
import { loadBundled } from "@amsintelligence/iconic";

const iconic = await loadBundled();
iconic.predict("go for a swim");
// [
//   { family: "swimming-pool", icon: "swimming-pool", score: 0.55 },
//   { family: "person-simple-swim", icon: "person-simple-swim", score: 0.25 },
//   { family: "beach-ball", icon: "beach-ball", score: 0.04 }
// ]
```

## Use in the browser

Host the files from the `model` folder next to your app, then load them by URL:

```js
import { Iconic } from "@amsintelligence/iconic";

const iconic = await Iconic.load("/iconic-model");
const [best] = iconic.predict("call the dentist");
console.log(best.icon); // "tooth"
```

The folder layout is the same as the Hugging Face repo, so you can also point
`Iconic.load` at a copy of that. In total the files are about 6 MB.

## What you get back

`predict(text, { k = 3 })` returns the `k` best matches, best first:

- `icon`: the Phosphor icon name to show
- `family`: the group of similar icons it came from
- `score`: how confident the model is, from 0 to 1

A prediction takes well under a millisecond.

## Tests

```bash
npm test
```

The tests check that tokens and features match the reference implementation
the model was trained with.

## License

Apache 2.0. Copyright 2026 Offchain Studio.
