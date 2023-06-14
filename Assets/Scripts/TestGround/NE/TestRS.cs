using Algorithms.NE;
using NN;
using TestGround.NE;
using UnityEngine;

public class TestRS : TestES
{
    protected override void Start()
    {
        if (populationSize % 2 != 0)
        {
            populationSize++;
        }

        _env.CreatePopulation(populationSize);
        _currentSates = _env.DistributedResetEnv();


        var network = new NetworkLayer[]
        {
            new ESNetworkLayer(AlgorithmNE.RS, populationSize, noiseStandardDeviation, _env.GetObservationSize, 128,
                ActivationFunction.ReLu, Instantiate(shader), 25f, true),
            new ESNetworkLayer(AlgorithmNE.RS, populationSize, noiseStandardDeviation, 128, 128,
                ActivationFunction.ReLu, Instantiate(shader), 25f, true),
            new ESNetworkLayer(AlgorithmNE.RS, populationSize, noiseStandardDeviation, 128, _env.GetNumberOfActions,
                ActivationFunction.Linear, Instantiate(shader), 25f, true)
        };

        var neModel = new ESModel(network);
        _neModel = new RS(neModel, _env.GetNumberOfActions, populationSize);

        Time.timeScale = simulationSpeed;
    }
}
