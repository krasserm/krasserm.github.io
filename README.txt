## Initial setup

- Install Jekyll on Ubuntu: https://jekyllrb.com/docs/installation/ubuntu/
- gem install github-pages
- npm install grunt --save-dev
- brew install pandoc
- Install pdflatex (https://gist.github.com/yspkm/f33d59181b7f6f5c8701360995c07418)

## Other

- Generate resume.pdf: pandoc resume.md -o resume.pdf
- Generate .css from .less files by running: grunt
- Generate resume.pdf by running: pandoc -o resume.pdf resume.md
- Uses chruby as version manager (uses 3.4.1)
