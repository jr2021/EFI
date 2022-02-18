from EASI import EASI
import sys
import pickle

def main():
    instance = EASI(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    instance.genetic_algorithm()
    with open('test.pkl', 'wb') as file:
        pickle.dump(instance, file)

main()
