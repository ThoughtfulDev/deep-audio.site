# deep-audio.site

**TL;DR: Does your Song sound like Taylor Swift, Michael Jackson, Ed Sheeran or Ariana Grande? [Find out now](http://deep-audio.site)!**

Explanation is [here](http://deep-audio.site/about).

## How to

(Optional create a virtualenv)
```
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Clone the Repo & Install Dependencies
```
git clone https://github.com/thoughtfuldev/deep-audio.site
pip install tensorflow
pip install -r requirements.txt
```

**Ensure that you have ffmpeg installed and added to your path**
e.g
### Linux
`$ sudo apt install ffmpeg`
### Windows
Install some codecs or something... i have no Idea sry...

## Using
### Pretrained Model
If you don't want to train with your own Songs then you can just run 
```
$ python Web.py
```
and go to http://localhost:5000 upload you Song and get the output
**OR** use the cli e.g
```
python use.py <filename>
```

### Training your own Model
1. Add the Songs to the corresponding Folders in *.data/taylor_swift, ./data/ariana_grande* etc...
2. `$ python preprocess.py` (This will take some time)
3. `$ python nn_train.py` (to train the neural network)
4. `$ python svm_train.py` (to train the SVM)
5. `$ python Web.py OR python use.py <filename>`

# License
```
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2018 ThoughtfulDev

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
```
