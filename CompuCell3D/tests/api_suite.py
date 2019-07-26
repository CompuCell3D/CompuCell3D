# tests/runner.py
import unittest

# import your test modules
import test_api

if __name__=='__main__':
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(test_api))
    # suite.test_api(test_api.test_core_object_createion_api)
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)