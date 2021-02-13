class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'cihp':
            return './input/test/test/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
