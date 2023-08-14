using System;
using System.Collections.Generic;
using NN.CPU_Single;
using UnityEngine;

public class NEATNeworkModel
{
    public List<NEATNeuron> neurons;
    public List<Links> neuronLinks;


    public float CompatabilityScore(NEATNeworkModel networkToCompare)
    {
        float disjointNumber = 0;
        float excessNumber = 0;
        float matchedNumber = 0;
        float WeightDifference = 0;

        int genomeIndex1 = 0;
        int genomeIndex2 = 0;

        while (genomeIndex1 < neuronLinks.Count - 1 || genomeIndex2 < networkToCompare.neuronLinks.Count - 1)
        {
            if (genomeIndex1 == neuronLinks.Count - 1)
            {
                ++genomeIndex2;
                ++excessNumber;
                continue;
            }

            if (genomeIndex2 == networkToCompare.neuronLinks.Count - 1)
            {
                ++genomeIndex1;
                ++excessNumber;
                continue;
            }

            int linkId1 = neuronLinks[genomeIndex1].inovationNumber;
            int linkId2 = networkToCompare.neuronLinks[genomeIndex2].inovationNumber;

            if (linkId1 == linkId2)
            {
                ++genomeIndex1;
                ++genomeIndex2;
                ++matchedNumber;

                WeightDifference += Math.Abs(neuronLinks[genomeIndex1].weight -
                                             networkToCompare.neuronLinks[genomeIndex2].weight);
            }

            if (linkId1 < linkId2)
            {
                ++disjointNumber;
                ++genomeIndex1;
            }

            if (linkId1 > linkId2)
            {
                ++disjointNumber;
                ++genomeIndex2;
            }
        }

        int longestNetwork = networkToCompare.neuronLinks.Count;
        if (neuronLinks.Count > longestNetwork)
        {
            longestNetwork = neuronLinks.Count;
        }

        const float disjointWeight = 1;
        const float excessWeight = 1;
        const float matchedWeight = 0.4f;

        var score = (excessNumber * excessWeight / longestNetwork) +
                    (disjointNumber * disjointWeight / longestNetwork) +
                    (WeightDifference * matchedWeight / matchedNumber);
        return score;
    }
}

public class NEATNeuron
{
    public List<Links> neuronLinks;
}

public class Links
{
    public int inovationNumber;
    public float weight;
}