import numpy as np
import elder_futhark as ef


def get_data_sets():
    """
    This function creates two other functions that generate data.
    One generates a training data set and the other, an evaluation set.

    The examples have the format of a two-dimensional numpy array.
    They can be thought of as a very small (7-pixel by 7-pixel) image.

    The examples are drawn from the 24-rune alphabet of Elder Futhark.


    To use in a script:

        import data_loader_nordic_runes as dat

        training_generator, evaluation_grenerator = dat.get_data_sets()
        new_training_example = training_generator.next()
        new_evaluation_example = evaluation_generator.next()
    """

    examples = list(ef.runes.values())

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    return training_set, evaluation_set
