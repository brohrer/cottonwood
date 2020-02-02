# WE'VE MOVED

### [This repository is now being hosted on GitLab's servers.](https://gitlab.com/brohrer/cottonwood)
Check it out at [https://gitlab.com/brohrer/cottonwood](https://gitlab.com/brohrer/cottonwood).
Install from there. Update from there.

To make the switch on your local machine command line run

git remote set-url origin https://gitlab.com/brohrer/cottonwood.git

or

git remote set-url origin git@github.com:brohrer/cottonwood.git

depending on which protocol you're using.

This repo is being deprecated and will no longer be updated. BTW GitLab is pretty intuitive if you're already familiar with GitHub. [Check it out.](https://gitlab.com/)

--

--

--

--

# The Cottonwood Machine Learning Framework

Cottonwood is built to be as flexible as possible, top to bottom.
It's designed to minimize the iteration time when running experiments
and testing ideas. It's meant to be tweaked. Fork it. Add to it. Customize it
to solve the problem at hand. For more of the thought behind it, read
the post "
[Why another framework?](https://end-to-end-machine-learning.teachable.com/blog/171633/cottonwood-flexible-neural-network-framework)
and
[Why did you name it that?](https://end-to-end-machine-learning.teachable.com/blog/193739/why-is-it-called-cottonwood)

This code is always evolving. I recommend referencing a specific tag
whenever you use it in a project. Tags are labeled v1, v2, etc. and
the code attached to each one won't change.

If you want to follow along with the construction process for Cottonwood,
you can get a step-by-step walkthrough in the e2eML sequence
[Course 312](https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework/),
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/),
and
[Course 314](https://end-to-end-machine-learning.teachable.com/p/314-neural-network-optimization/).

## Installation

Whether you want to pull Cottonwood into another project, 
or experiment with ideas of your own, you'll want
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

Here is
[the cheatsheet for pulling the relevant components](cottonwood/doc/cheatsheet.md)
into your work.

## Versioning

Cottonwood versions are **not** guaranteed backward compatible.
You can select a particular version to work from.

```
cd cottonwood
git checkout v14
```

## Examples

See what Cottonwood looks like in action.
Feel free to use any of these as a template for a project of your own.
They're MIT licensed.

* [Compress images of the surface of Mars](
  https://github.com/brohrer/cottonwood_martian_images)

## Contribute to the project

[Here are some ideas to get you started.](cottonwood/doc/contributing.md)
