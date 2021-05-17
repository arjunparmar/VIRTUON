class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'cihp':
            return './model/input/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
