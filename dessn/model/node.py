import abc


class Node(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dependencies = []

    def add_dependency(self, dependency_class):
        self.dependencies.append(dependency_class)

    def get_dependencies(self):
        return self.dependencies

    @abc.abstractmethod
    def get_name(self):
        return

    @abc.abstractmethod
    def __call__(self):
        return
