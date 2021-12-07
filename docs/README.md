Before building the TF-COMB-documentation, please build tfcomb.

For building the documentation of tfcomb, add the packages of the ./required_packages.txt to your environment.
```
$ cd ./docs/
$ mamba install --file ./required_packages.txt
```  

Build documentation:

`make html`

Requires packages:
- tfcomb
- sphinx
- nbsphinx
- nbsphinx-link
- sphinx_rtd_theme
- pandoc
