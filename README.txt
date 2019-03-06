This file contains instructions to run the file flip.py to generate results.

At first make sure python 2.7 is installed in your system.

OS - Ubuntu 16.04

1 . sudo apt-get install virtualenv

2 . virtualenv testenv

#this will create the virtualenv needed for the experiment

3. source testenv/bin/activate

#this will activate the virtualenv

4. pip install -r requirements.txt

5. Run python flip.py 1 2 3

1 - images folder path (Default - 'images/')

2 - training_set.csv path (Default - 'training_set.csv')

3 - test.csv path (Default - 'test.csv')

1,2,3 are command line arguments.

Allowed configurations -
python flip.py (training_set.csv , test.csv and images/ in the same folder)
python flip.py images_path (training_set.csv and test.csv in the same folder)
python flip.py images_path training_set.csv_path (test.csv in the same folder)
python flip.py images_path training_set.csv_path testing.csv_path

6. After completely running through all and training all the models it will generate a "result.csv" file in the same directory.
That will be the submission file.
