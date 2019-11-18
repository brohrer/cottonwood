# The Cottonwood Machine Learning Framework

Cottonwood is built to as flexible as possible, top to bottom.
It's designed to minimize the iteration time when running experiments
and testing ideas. It's meant to be tweaked. Fork it. Add to it. Customize it
to solve the problem at hand. For more of the thought behind it, read
the post "
[Why another framework?](https://end-to-end-machine-learning.teachable.com/blog/171633/cottonwood-flexible-neural-network-framework)

This code is always evolving. I recommend referencing a specific tag
whenever you use it in a project. Tags are labeled v1, v2, etc. and
the code attached to each one won't change.

If you want to follow along with the construction process for Cottonwood,
you can get a step-by-step walkthrough in End-to-End Machine Learning
[Course 312](https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework/)
and
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/)

## Installation

If you'd like to experiment with ideas of your own, you'll want
to clone the repository to your local machine and install it from there.

```bash
git clone https://github.com/brohrer/cottonwood.git
python3 -m pip install -e cottonwood
```

## Try it out

```bash
python3
```
```python3
>>> import cottonwood.demo
```

## Versioning

Cottonwood versions are **not** guaranteed backward compatible.
You can select a particular version to work from.

```
cd cottonwood
git checkout v9
```

Check out the [Revision History](cottonwood/doc/revision_history.md)
to see what features and bugfixes come with each release.

## Examples

See what Cottonwood looks like in action.
Feel free to use any of these as a template for a project of your own.
They're MIT licensed.

* [Compress images of the surface of Mars](
  https://github.com/brohrer/cottonwood_martian_images)
