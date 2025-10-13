# Installation Guide

## 1-ignorelist
```
.gitignore

# Jekyll generated files
.jekyll-cache
.jekyll-metadata
.sass-cache
_asset_bundler_cache
_site
README.md
# Sublime Text
*.sublime-project
*.sublime-workspace

# Ruby Gem
*.gem
# .bundle
Gemfile.lock
**/vendor/bundle
```

## 2-install Git
install git 
connect to this project repository
pull all the files except for the ignored list above

## 3-ruby enviroment

- install rbenv to select ruby version
- use ruby 2.6.10-1 or system ruby in mac os (2.6.0-2.6.10)
- rbenv global 2.6.10-1      

## bundle install  
bundle exec jekyll serve --trace


