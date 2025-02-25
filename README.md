<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo-dark.png">
    <img alt="logo" src="img/logo-light.png" width="500px">
  </picture>
  <br>
  Use LLMs for document ranking.
</p>

## Description

There's power in AI in that you can "throw a problem at it" and get some result, without even fully defining the problem. For example, give it a bunch of code diffs and a security advisory, and ask, "Which of these diffs seems most likely to fix the security bug?" However, it's not always that easy:
- nondeterminism: doesn't always respond with the same result
- context window: can't pass in all the data at once, need to break it up
- output contraints: sometimes doesn't return all the data you asked it to review
- subjectivity in scoring: has a really hard time assigning a numeric score to an individual item

We built raink to circumvent those issues and solve general ranking problems that are otherwise difficult for LLMs to process. See our blog post [raink: Use LLMs for Document Ranking](https://bishopfox.com/blog/raink-llms-document-ranking) for more background on this technique, and our talk [Patch Perfect: Harmonizing with LLMs to Find Security Vulns](https://www.youtube.com/watch?v=IBuL1zY69tY) to see how we've applied raink to offensive security problems.

## Getting started

### Install

```
git clone https://github.com/noperator/raink
cd raink
go install
```

### Configure

Set your `OPENAI_API_KEY` environment variable.

### Usage

```
raink  -h
Usage of raink:
  -dry-run
    	Enable dry run mode (log API calls without making them)
  -encoding string
    	Tokenizer encoding (default "o200k_base")
  -f string
    	Input file
  -o string
    	JSON output file
  -ollama-model string
    	Ollama model name (if not set, OpenAI will be used)
  -ollama-url string
    	Ollama API URL (default "http://localhost:11434/api/chat")
  -openai-model string
    	OpenAI model name (default "gpt-4o-mini")
  -p string
    	Initial prompt (prefix with @ to use a file)
  -r int
    	Number of runs (default 10)
  -ratio float
    	Refinement ratio as a decimal (e.g., 0.5 for 50%) (default 0.5)
  -s int
    	Number of items per batch (default 10)
  -t int
    	Max tokens per batch (default 128000)
  -template string
    	Template for each object in the input file (prefix with @ to use a file) (default "{{.Data}}")

```

Compares 100 [sentences](https://github.com/noperator/raink/blob/main/testdata/sentences.txt) in under 2 min.

```
raink \
    -f testdata/sentences.txt \
    -r 10 \
    -s 10 \
    -p 'Rank each of these items according to their relevancy to the concept of "time".' |
    jq -r '.[:10] | map(.value)[]' |
    nl

   1  The train arrived exactly on time.
   2  The old clock chimed twelve times.
   3  The clock ticked steadily on the wall.
   4  The bell rang, signaling the end of class.
   5  The rooster crowed at the break of dawn.
   6  She climbed to the top of the hill to watch the sunset.
   7  He watched as the leaves fell one by one.
   8  The stars twinkled brightly in the clear night sky.
   9  He spotted a shooting star while stargazing.
  10  She opened the curtains to let in the morning light.
```

#### JSON Support

If the input file is a JSON document, it will be read as an array of objects and each object will be used for ranking.

For instance, two objects would be loaded and ranked from this document:

```json
[
  {
    "path": "/foo",
    "code": "bar",
  },
  {
    "path": "/baz",
    "code": "nope",
  }
]
```

#### Templates

It is possible to include each element from the input file in a template using the [golang template syntax](https://pkg.go.dev/text/template) via the `-template "template string"` (or `-template @file.tpl`) argument.

For text input files, each line can be referenced in the template with the `Data` variable:

```
Anything you want with {{ .Data }}
```

For JSON input files, each object in the array can be referenced directly. For instance, elements of the previous JSON example can be referenced in the template code like so:

```
# {{ .path }}

{{ .code }}
```

## Back matter

### See also

- [raink: Use LLMs for Document Ranking](https://bishopfox.com/blog/raink-llms-document-ranking)
- [Patch Perfect: Harmonizing with LLMs to Find Security Vulns](https://www.youtube.com/watch?v=IBuL1zY69tY)
- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/html/2306.17563v2)
- [Introducing Rerank 3.5: Precise AI Search](https://cohere.com/blog/rerank-3pt5)

### To-do

- [x] parallelize openai calls for each run
- [x] save time by using shorter hash ids
- [x] make sure that each randomized run is evenly split into groups so each one gets included/exposed
- [ ] allow specifying an input _directory_ (where each file is distinct object)
- [x] alert if the incoming context window is super large
- [x] some batches near the end of a run (9?) are small for some reason
- [ ] run openai batch mode
- [x] automatically calculate optimal batch size?
- [x] explore "tournament" sort vs complete exposure each time
- [ ] add parameter for refinement ratio
- [ ] add blog link
- [ ] support non-OpenAI models
- [ ] add boolean refinement flag

### License

This project is licensed under the [MIT License](LICENSE).
