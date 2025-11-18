# Installation Guide

https://github.com/mmistakes/minimal-mistakes/tree/master

## 1-ignorelist
```
#忽略gitignore文件
.gitignore

# macOS
.DS_Store
.ruby-version

# Jekyll generated files
.jekyll-cache
.jekyll-metadata
.sass-cache
_asset_bundler_cache
_site
README.md


# Ruby Gem
#_config.yml
*.gem
.bundle
Gemfile.lock
**/vendor/bundle
#Gemfile


# Jekyll generated files
_site
.jekyll-cache
.jekyll-metadata
.sass-cache
_asset_bundler_cache
.jekyll-cache/Jekyll/Cache/
# /vendor/
# vendor/
# /vendor


# Sublime Text
*.sublime-project
*.sublime-workspace


#  .gitignore文件实例：

# *.a       # 忽略所有 .a 结尾的文件
# !lib.a    # 但 lib.a 除外
# /TODO     # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
# build/    # 忽略 build/ 目录下的所有文件
# doc/*.txt # 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
#

# Notice1！：lock the two following files in _site
## otherwise, it will be deleted everytime we bundle exec jekyll serve
package-lock.json
package.json


# Notice2！：
when try to "bundle install"
need to delete anaconda/openssl in the system path, at least temporarily!

```

## 2-install Git
install git 
connect to this project repository
pull all the files except for the ignored list above

## 3-ruby enviroment

- install rbenv to select ruby version
- use ruby 2.6.10-1 or system ruby in mac os (2.6.0-2.6.10)
- rbenv global 2.6.10-1      

## build the web

```
bundle install  
bundle exec jekyll serve --trace
```
## notice

In github theme, lower case and upper case in file names will **make a difference**.
