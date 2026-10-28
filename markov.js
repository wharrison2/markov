// Browser port of ../generate.py: a word-level backoff Markov chain, loaded pre-trained from model.json.

// 32-bit FNV-1a hash of the UTF-8 bytes; build.py computes the same thing.
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) h = Math.imul(h ^ byte, 0x01000193) >>> 0;
  return h;
}

function sample(followers, rng, weightOf = (word, w) => w) {
  let total = 0;
  for (const [word, w] of followers) total += weightOf(word, w);
  let r = rng() * total;
  for (const [word, w] of followers) {
    if ((r -= weightOf(word, w)) < 0) return word;
  }
  return followers.keys().next().value;
}

// Similarity ratio matching Python's difflib.SequenceMatcher(None, a, b).ratio() for short strings.
function matchingChars(a, b) {
  if (!a.length || !b.length) return 0;
  let best = 0, bi = 0, bj = 0;
  let prev = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1;
        if (cur[j] > best) [best, bi, bj] = [cur[j], i - cur[j], j - cur[j]];
      }
    }
    prev = cur;
  }
  if (!best) return 0;
  return best + matchingChars(a.slice(0, bi), b.slice(0, bj)) +
    matchingChars(a.slice(bi + best), b.slice(bj + best));
}

function correct(word, model, vocab) {
  // Return the most similarly spelled word we've seen (e.g. fixing a typo), or the word itself.
  const lower = word.toLowerCase();
  if (model.has(lower)) return word;
  let best = null, bestScore = 0.7;
  for (const candidate of vocab) {
    const total = lower.length + candidate.length;
    // Cheap upper bound before the full comparison, like difflib's real_quick_ratio.
    if ((2 * Math.min(lower.length, candidate.length)) / total < bestScore) continue;
    const score = (2 * matchingChars(lower, candidate)) / total;
    if (score > bestScore || (score === bestScore && best !== null && candidate > best)) {
      [best, bestScore] = [candidate, score];
    }
  }
  return best ?? word;
}

function nextWord(model, history, maxOrder, rng, weightOf) {
  // Pick the next word using the longest context we've seen, backing off to shorter ones.
  for (let n = Math.min(maxOrder, history.length); n > 0; n--) {
    const key = history.slice(-n).map((w) => w.toLowerCase()).join(" ");
    const followers = model.get(key);
    if (followers) return sample(followers, rng, weightOf);
  }
  return null;
}

export function mulberry32(seed) {
  return function () {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Chain {
  constructor(data) {
    // Contexts arrive as flat [word index, weight, ...] arrays; unpack them into follower Maps.
    this.model = new Map();
    for (const [context, flat] of Object.entries(data.contexts)) {
      const followers = new Map();
      for (let i = 0; i < flat.length; i += 2) followers.set(data.words[flat[i]], flat[i + 1]);
      this.model.set(context, followers);
    }
    this.vocab = [...this.model.keys()].filter((k) => !k.includes(" "));
    this.maxOrder = data.maxOrder;
    this.quoteStreak = data.quoteStreak;
    this.quoteHashes = new Set(data.quoteHashes);
    this.periodRamp = data.periodRamp;
    // Drop closing quotes/brackets so "food.”" still counts as ending in a period.
    this.stripClosers = (word) => {
      let end = word.length;
      while (end && data.closers.includes(word[end - 1])) end--;
      return word.slice(0, end);
    };
    // How often each word appears, for jumping somewhere random at a dead end.
    this.wordWeights = new Map();
    for (const context of this.vocab) {
      for (const [word, weight] of this.model.get(context)) {
        this.wordWeights.set(word, (this.wordWeights.get(word) || 0) + weight);
      }
    }
  }

  periodBoost(sentenceWords) {
    // Same ramp as period_boost() in generate.py.
    let boost = 1;
    this.periodRamp.forEach(([start, step], i) => {
      const end = i + 1 < this.periodRamp.length ? this.periodRamp[i + 1][0] : sentenceWords;
      boost += step * Math.max(0, Math.min(sentenceWords, end) - start);
    });
    return boost;
  }

  generate(prompt, length, maxOrder, rng) {
    const { model, vocab } = this;
    maxOrder = Math.min(maxOrder, this.maxOrder);
    // Generate as if the prompt's unseen words were spelled like their closest matches in the corpus.
    const history = prompt.split(/\s+/).filter(Boolean).map((w) => correct(w, model, vocab));
    // The last word is one we've never seen, with no close spelling either: give up rather than guess.
    if (history.length && !model.has(history.at(-1).toLowerCase())) return null;
    const output = [];
    const endsSentence = (w) => /[.!?]$/.test(this.stripClosers(w));
    // Words in the current sentence so far, counting any unfinished sentence at the end of the prompt.
    let sentenceWords = history.length - 1 - history.findLastIndex(endsSentence);
    for (let i = 0; i < length; i++) {
      // As the sentence runs long, words ending in a period get a growing weight multiplier.
      const boost = this.periodBoost(sentenceWords);
      const weightOf = (word, w) => (/\.$/.test(this.stripClosers(word)) ? w * boost : w);
      const tail = history.slice(-this.quoteStreak).map((w) => w.toLowerCase()).join(" ");
      const quoting = history.length >= this.quoteStreak && this.quoteHashes.has(fnv1a(tail));
      let word = nextWord(model, history, quoting ? Math.max(1, maxOrder - 1) : maxOrder, rng, weightOf);
      if (word === null) {
        // Dead end (or prompt words never seen): jump to a random word from the corpus.
        word = sample(this.wordWeights, rng, weightOf);
      }
      output.push(word);
      history.push(word);
      sentenceWords = endsSentence(word) ? 0 : sentenceWords + 1;
    }
    return output.join(" ");
  }
}
