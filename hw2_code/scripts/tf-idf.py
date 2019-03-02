import sys
import numpy
import pickle


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit(1)

    model_feat = pickle.load(open(sys.argv[1], 'rb'))
    sz = int(sys.argv[2])

    df = numpy.zeros(sz)
    for key in model_feat:
        df += model_feat[key]

    # performing tf-idf on each histogram vector of dimension k
    total_frames = numpy.sum(df)

    df[df == 0] = 1
    for id in model_feat:
        model_feat[id] *= numpy.log(total_frames / df)

    file_name = sys.argv[1].split(".")[0]
    file_name += "_idf.pickle"
    print(file_name)
    with open(file_name, 'wb') as w:
        pickle.dump(model_feat, w)