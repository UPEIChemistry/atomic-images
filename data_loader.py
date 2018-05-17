from keras_model_loader import kml

import os
import numpy as np


try:
    xrange
except NameError:
    xrange = range


if os.getenv('USER') == 'trevor':
    RUNNING_ON = 'trevor-pc'
else:
    RUNNING_ON = 'cedar'
PATHS = {
    'trevor-pc': {
        'qm9_data': '/home/trevor/Documents/GDrive/Data/QM9_data/qm9_data_loader.py'
    },
    'cedar': {
        'qm9_data': '/home/trevorpe/jpearson_storage/Data/QM9_data/qm9_data_loader.py'
    }
}
data_loader_template = PATHS[RUNNING_ON]['qm9_data']
qm9_template = kml.import_module(data_loader_template)

bins_file = kml.register_extra_file('split_indices.npz', name='split_indices')
atom_refs_file = kml.register_extra_file('atom_refs.npz', name='atom_refs')


class DataLoader(qm9_template.QM9DataLoader):
    def __init__(self, **kwargs):
        """
        Constructor for the data loader. Make sure to call the super function.
        """
        super(DataLoader, self).__init__(keys=['U_naught', 'R', 'Z'])

        self.bin_indices = None
        self.val_bin = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.atom_ref = None
        self.atom_refs_property = 'U0'

    def get_default_parameters(self, **kwargs):
        old = super(DataLoader, self).get_default_parameters(**kwargs)
        old.update({
            'test_set_fraction': 0.2,
            'n_bins': 5
        })
        return old

    def load_data(self, val_bin=None, **kwargs):
        super(DataLoader, self).load_data(**kwargs)
        self.coordinates = self.data['R']
        self.coordinates[np.isnan(self.coordinates)] = 0.0

        self.atomic_numbers = self.data['Z'].astype(np.uint8)
        self.u_naught = self.data['U_naught']

        in_data = [self.atomic_numbers, self.coordinates]
        out_data = [self.u_naught]

        if self.atom_ref is None and os.path.exists(atom_refs_file):
            print('Loading atom reference energies from "%s"' % atom_refs_file)
            atom_ref = np.load(atom_refs_file)

            labels = atom_ref['labels']
            for i, v in enumerate(labels):
                if v == self.atom_refs_property:
                    self.atom_ref = atom_ref['atom_ref'][:, i]
                    break
            self.atom_ref = np.squeeze(self.atom_ref)
            max_atomic_number = np.max(self.atomic_numbers)
            self.atom_ref = self.atom_ref[:max_atomic_number + 1]
            print('Converting eV to Hartrees')
            self.atom_ref /= 27.21138602  # eV -> Ha

        if val_bin is None:
            val_bin = 0

        if self.bin_indices is None:
            if os.path.exists(bins_file):
                print('Loading splits from "%s"' % bins_file)
                self.bin_indices = np.load(bins_file)
            else:
                print('Splitting datasets...')
                sorted_args = np.argsort(out_data[0])

                i_test = int(1 / self.parameters['test_set_fraction'])
                n_bins = self.parameters['n_bins']
                n_total = out_data[0].shape[0]

                # Split test set
                test_bin = np.array(
                    [i for i in xrange(n_total) if i % i_test == 0],
                    dtype=np.int32
                )
                test_bin = sorted_args[test_bin]
                np.random.shuffle(test_bin)

                train_val_bin = np.array(
                    [i for i in xrange(n_total) if i % i_test != 0],
                    dtype=np.int32
                )
                train_val_bin = sorted_args[train_val_bin]
                np.random.shuffle(train_val_bin)

                # Pad with -1
                train_val_remainder = train_val_bin.shape[0] % n_bins
                train_val_bin = np.pad(
                    train_val_bin,
                    (0, n_bins - train_val_remainder),
                    'constant',
                    constant_values=-1
                )
                train_val_bin = np.reshape(train_val_bin, (n_bins, -1))

                self.bin_indices = {
                    'test': test_bin,
                    'train': train_val_bin
                }

                print('Saving splits to "%s"' % bins_file)
                np.savez_compressed(bins_file,
                                    **self.bin_indices)

            # Get test data
            test_indices = self.bin_indices['test']
            self.X_test = [item[test_indices] for item in in_data]
            self.y_test = [item[test_indices] for item in out_data]

        if self.val_bin != val_bin:
            not_val_bins = [i for i in xrange(self.bin_indices['train'].shape[0])
                            if i != val_bin]
            val_indices = self.bin_indices['train'][val_bin]
            train_indices = np.concatenate(self.bin_indices['train'][not_val_bins, :])

            # Mask -1 indices since they are padding
            train_indices = train_indices[train_indices != -1]
            val_indices = val_indices[val_indices != -1]

            # Assign data
            # FIXME: remove testing slice
            self.X_train = [item[train_indices][0:1000] for item in in_data]
            self.y_train = [item[train_indices][0:1000] for item in out_data]

            # Compute y_mean and y_sigma
            n_atoms = np.sum(self.X_train[0] != 0, axis=1)
            if self.atom_ref is not None:
                one_hot_z = np.eye(self.atom_ref.shape[0])[self.X_train[0]]
                y_train_ref = np.dot(np.sum(one_hot_z, axis=1), self.atom_ref)
                del one_hot_z

                e_diff_per_atom = (self.y_train[0] - y_train_ref) / n_atoms

                self.y_diff_mu_per_atom = np.mean(e_diff_per_atom, axis=0)
                self.y_diff_sigma_per_atom = np.std(e_diff_per_atom, axis=0)
            e_per_atom = self.y_train[0] / n_atoms
            self.y_mu_per_atom = np.mean(e_per_atom, axis=0)
            self.y_sigma_per_atom = np.std(e_per_atom, axis=0)
            self.y_mu = np.mean(self.y_train[0])
            self.y_sigma = np.std(self.y_train[0])
            del n_atoms

            self.X_val = [item[val_indices][0:1000] for item in in_data]
            self.y_val = [item[val_indices][0:1000] for item in out_data]

            self.X_shape = [item.shape[1:] if len(item.shape) > 1 else (1,)
                            for item in self.X_train]
            self.y_shape = [item.shape[1:] if len(item.shape) > 1 else (1,)
                            for item in self.y_train]

            self.val_bin = val_bin

    def get_cross_validation_k(self, **kwargs):
        """Return the number of folds for k-fold cross-validation

        Args:
            training_index (int, optional): iteration counter if
                optimizing
            **kwargs: remaining keyword arguments

        Returns:
            int: number of folds to run when cross-validating
        """
        return self.parameters['n_bins']

    def get_training_data(self, **kwargs):
        """Called to obtain the training data. Should either return a
        tuple of numpy arrays containing the full dataset as
        (X_training_data, Y_training_data) or yield batches of
        training data in a tuple (batch_X, batch_Y)

        Args:
            training_index (int, optional): iteration counter if
                optimizing
            **kwargs: remaining keyword arguments

        Returns:
            generator or tuple: training data as numpy arrays
        """
        return self.X_train, self.y_train

    def get_validation_data(self, **kwargs):
        """Called to obtain the validation data. Should either return a
        tuple of numpy arrays containing the full dataset as
        (X_validation_data, Y_validation_data) or yield batches of
        validation data in a tuple (batch_X, batch_Y)

        Args:
            training_index (int, optional): iteration counter if
                optimizing
            **kwargs: remaining keyword arguments

        Returns:
            generator or tuple: validation data as numpy arrays
        """
        return self.X_val, self.y_val

    def get_testing_data(self, **kwargs):
        """Called to obtain the testing data. Should either return a
        tuple of numpy arrays containing the full dataset as
        (X_testing_data, Y_testing_data) or yield batches of
        testing data in a tuple (batch_X, batch_Y)

        Args:
            training_index (int, optional): iteration counter if
                optimizing
            **kwargs: remaining keyword arguments

        Returns:
            generator or tuple: testing data as numpy arrays
        """
        return self.X_test, self.y_test
