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

We built raink to circumvent those issues and solve general ranking problems that are otherwise difficult for LLMs to process. See [Patch Perfect: Harmonizing with LLMs to Find Security Vulns](https://www.youtube.com/watch?v=IBuL1zY69tY) for more background.

## Getting started

### Install

```
git clone https://github.com/bishopfox/raink
cd raink
go install
```

### Configure

Set your `OPENAI_API_KEY` environment variable.

### Usage

```
raink  -h
Usage of raink:
  -f string
    	Input file
  -p string
    	Initial prompt
  -r int
    	Number of runs (default 10)
  -s int
    	Batch size (default 10)
```

Compares 100 [sentences](https://github.com/BishopFox/raink/blob/main/testdata/sentences.txt) in under 2 min.

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

## Back matter

### See also

- https://cohere.com/blog/rerank-3pt5
- https://www.youtube.com/watch?v=IBuL1zY69tY

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

### License

This project is licensed under the [MIT License](LICENSE).
