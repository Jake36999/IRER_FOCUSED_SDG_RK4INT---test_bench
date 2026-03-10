import unittest

class TestFSSScalingAnalyzer(unittest.TestCase):
    def test_simulation_with_new_parameters(self):
        # Simulate with new parameters
        result = run_simulation(param_D=4.739, param_splash_coupling=0.15, param_splash_fraction=-0.5)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)

    def test_output_integrity(self):
        # Check output against expected results
        output = analyze_output('rho_history_true_golden.h5')
        expected_output = load_expected_output()
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()