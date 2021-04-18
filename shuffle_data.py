import os, random, shutil, sys, shutil
from distutils.dir_util import copy_tree

from numpy.lib.twodim_base import _trilu_indices_form_dispatcher

base_dir = 'data'
true_dir = 'Autistic'
false_dir = 'Non_Autistic'

def MoveRandom(source, dest, no_of_files):

    #Prompting user to enter number of files to select randomly along with directory
    # source=input("Enter the Source Directory : ")
    # dest=input("Enter the Destination Directory : ")
    # no_of_files=int(input("Enter The Number of Files To Select : "))

    print("%"*25+"{ Details Of Transfer }"+"%"*25)
    print("\n\nList of Files Moved from %s to %s :-" % (source, dest))

    #Using for loop to randomly choose multiple files
    for i in range(no_of_files):
        #Variable random_file stores the name of the random file chosen
        random_file=random.choice(os.listdir(source))
        print("%d} %s"%(i+1,random_file))
        source_file="%s/%s"%(source,random_file)
        dest_file=dest
        #"shutil.move" function moves file from one directory to another
        shutil.move(source_file,dest_file)

    print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        valid_amount = int(sys.argv[1])
        test_amount = 0
    elif len(sys.argv) == 3:
        valid_amount = int(sys.argv[1])
        test_amount = int(sys.argv[2])
    else:
        print("Input1: amount of images for validation.")
        print("Input2: amount of images for testing (defalut is 0).")
        sys.exit()

    if not os.path.isdir(base_dir+'/consolidated'):
        os.mkdir(base_dir+'/consolidated')
        copy_tree(base_dir+'/train', base_dir+'/consolidated')
        copy_tree(base_dir+'/valid', base_dir+'/consolidated')
        copy_tree(base_dir+'/test', base_dir+'/consolidated')

    shutil.rmtree(base_dir+'/train', ignore_errors=True)
    shutil.rmtree(base_dir+'/valid', ignore_errors=True)
    shutil.rmtree(base_dir+'/test', ignore_errors=True)

    shutil.copytree(base_dir+'/consolidated', base_dir+'/train')

    os.makedirs(base_dir+'/valid/'+true_dir)
    os.makedirs(base_dir+'/valid/'+false_dir)
    MoveRandom(base_dir+'/train/'+true_dir, base_dir+'/valid/'+true_dir, valid_amount)
    MoveRandom(base_dir+'/train/'+false_dir, base_dir+'/valid/'+false_dir, valid_amount)

    os.makedirs(base_dir+'/test/'+true_dir)
    os.makedirs(base_dir+'/test/'+false_dir)
    MoveRandom(base_dir+'/train/'+true_dir, base_dir+'/test/'+true_dir, test_amount)
    MoveRandom(base_dir+'/train/'+false_dir, base_dir+'/test/'+false_dir, test_amount)


