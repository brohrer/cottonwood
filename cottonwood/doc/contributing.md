# Contributing

Feeling inspired to contribute to the Cottonwood community? Fantastic!

Here are some ideas:

### Use it to build something

The single most powerful thing you can do is show someone else what
they can accomplish with it. For example

* Predict the temperature in your hometown

* Perform optical character recognition

* Flag potential grammar mistakes in text

Then when you’re done, share the code and a description of how to use it.
This will help newcomers to get up and going quickly, and and it’s a
great informal publication for your resume.

### Try out your new ideas

Cottonwood is intended to be easy to experiment with. Have a new
funky layer design you’d like to test drive? Cottonwood is the
perfect tool for prototyping it and seeing if you like the results.

### Build out the code

Cottonwood is still in its infancy. It has only a small fraction of
the features in more mature frameworks. If you would like to add a
commonly used activation function or layer type that’s
missing, go for it! Some things to keep in mind:

* Above all else, Cottonwood is intended to maximize flexibility and
simplicity. To this end, the `core` code avoids unnecessary
libraries and constructions, even at the expense of performance.
Keep your code simple and as straightforward as you can.

* The `core` subpackage is for canonical tools and techniques that have
been published and have demonstrated their usefulness.

* The `experimental` subpackage is for tools and techniques that haven’t
yet entered the mainstream. If you have something you are trying
out and would like to share it around, `experimental` is the place for it.

The workflow for submitting code that gives the clearest paper trail
is

1) open an issue describing what you plan to submit and why
([here is a stellar example](https://github.com/brohrer/cottonwood/issues/2)),

2) write the code,

3) test it in an example,

3) submit your changes in a pull request. That will give us a chance
to look it over to together and discuss it if necessary,

4) after that it will be merged into the main code base.
