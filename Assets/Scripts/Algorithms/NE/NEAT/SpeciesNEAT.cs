using System;
using System.Collections.Generic;
using Random = UnityEngine.Random;

namespace Algorithms.NE.NEAT
{
    public class SpeciesNEAT
    {
        private GenomeNEAT _leader;
        private readonly int _speciesId;
        private readonly int _youngAgeThreshold;
        private readonly float _youngFitnessBonus;
        private readonly int _oldAgeThreshold;
        private readonly float _oldFitnessPenalty;

        private readonly List<GenomeNEAT> _members;

        private float _bestFitnessSoFar;
        private float _currentMinFitness;

        private int _generationsSinceImprovement;
        private int _speciesAge;

        public GenomeNEAT Leader => _leader;

        public int GenerationsSinceImprovement => _generationsSinceImprovement;
        public float LeaderFitness => Leader.Fitness;
        public float BestFitnessSoFar => _bestFitnessSoFar;
        public int SpecieMemberCount => _members.Count;

        public SpeciesNEAT(GenomeNEAT firstElement, int speciesId, int youngAgeThreshold, float youngFitnessBonus,
            int oldAgeThreshold, float oldFitnessPenalty)
        {
            _leader = firstElement;
            _speciesId = speciesId;
            _youngAgeThreshold = youngAgeThreshold;
            _youngFitnessBonus = youngFitnessBonus;
            _oldAgeThreshold = oldAgeThreshold;
            _oldFitnessPenalty = oldFitnessPenalty;
            _members = new List<GenomeNEAT> { firstElement };

            _bestFitnessSoFar = float.MinValue;
            _generationsSinceImprovement = 0;
            _speciesAge = 0;
        }

        public void AddMember(GenomeNEAT newMember)
        {
            //TODO: comparing Ids could be faster
            if (newMember == _leader) return;

            _members.Add(newMember);
            if (newMember.Fitness > _leader.Fitness)
            {
                _leader = newMember;
            }
        }

        public float AdjustedFitness()
        {
            _currentMinFitness = float.MaxValue;
            var total = 0f;
            var membersSize = _members.Count;
            for (int i = 0; i < membersSize; i++)
            {
                var member = _members[i];
                var fitness = member.Fitness;
                if (_speciesAge < _youngAgeThreshold)
                {
                    var bonus = Math.Abs(fitness * _youngFitnessBonus);
                    fitness += bonus;
                }
                else if (_speciesAge > _oldAgeThreshold)
                {
                    var penalty = Math.Abs(fitness * _oldFitnessPenalty);
                    fitness -= penalty;
                }

                var adjustedFitness = fitness / membersSize;
                total += adjustedFitness;
                member.AdjustedFitness = adjustedFitness;

                if (_currentMinFitness < member.Fitness) continue;
                _currentMinFitness = member.Fitness;
            }

            return total;
        }

        public float CalculateSpawnAmount(float populationAdjustedFitness)
        {
            var spawnAmount = 0f;
            for (int i = 0; i < _members.Count; i++)
            {
                spawnAmount += _members[i].AdjustedFitness / populationAdjustedFitness;
            }

            return spawnAmount;
        }
        
        public GenomeNEAT GetRandomMember(GenomeNEAT memberNotToUse = null)
        {
            float sum = 0;
            for (int i = 0; i < _members.Count; i++)
            {
                var member = _members[i];
                if(member == memberNotToUse) continue;
                
                sum += member.Fitness - _currentMinFitness + 1;
            }
            
            var randomProbability = Random.Range(1e-20f, sum);
            var index = -1;
            while (randomProbability > 0f)
            {
                var member = _members[++index];
                if(member == memberNotToUse) continue;
                
                //TODO: Might not go up to the sum value
                randomProbability -= member.Fitness - _currentMinFitness + 1;
            }

            return _members[index];
        }

        public void ResetSpecieMembers()
        {
            //TODO: might be better to make a copy of the leader instead of getting a reference
            if (_members.Count > 1)
            {
                _members.Clear();
                _members.Add(_leader);
            }

            if (_leader.Fitness > _bestFitnessSoFar)
            {
                _bestFitnessSoFar = _leader.Fitness;
                _generationsSinceImprovement = 0;
            }
            else
            {
                _generationsSinceImprovement++;
            }

            _speciesAge++;
        }
    }
}