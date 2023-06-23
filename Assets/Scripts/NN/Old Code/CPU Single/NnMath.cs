using UnityEngine;

namespace NN.CPU_Single
{
    public static class NnMath
    {
        public static float Sign(float value)
        {
            return value >= 0.0f ? 1f : -1f;
        }

        public static float Clamp(float value, float min, float max)
        {
            if (value < min)
            {
                value = min;
            }
            else if (value > max)
            {
                value = max;
            }

            return value;
        }

        public static float ArrayMax(float[] valueArray)
        {
            float num = valueArray[0];
            for (int i = 1; i < valueArray.Length; i++)
            {
                var currentVal = valueArray[i];
                if (num > currentVal) continue;

                num = currentVal;
            }

            return num;
        }

        public static float ArrayMean(float[] arr)
        {
            float result = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                result += arr[i];
            }

            return result / arr.Length;
        }

        //TODO: consider sending the result matrices as function parameters, so there is no need to create a new one
        // every time we run a function
        public static float[,] CopyMatrix(float[,] mat)
        {
            var copyMat = new float[mat.GetLength(0), mat.GetLength(1)];
            for (int i = 0; i < mat.GetLength(0); i++)
            {
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    copyMat[i, j] = mat[i, j];
                }
            }

            return copyMat;
        }

        public static bool CopyMatrix(float[,] inMat, float[,] matToCopy)
        {
            var isNan = false;
            int matRowSize = matToCopy.GetLength(1);
            for (int i = 0; i < matToCopy.GetLength(0); i++)
            {
                for (int j = 0; j < matRowSize; j++)
                {
                    // if (float.IsNaN( matToCopy[i, j]))
                    // {
                    //     isNan = true;
                    // }
                    inMat[i, j] = matToCopy[i, j];
                }
            }

            return isNan;
        }

        public static float[,] MatrixDotProduct(float[,] mat1, float[,] mat2)
        {
            var mat1ColumnSize = mat1.GetLength(0);
            var mat2RowSize = mat2.GetLength(1);

            var output = new float[mat1ColumnSize, mat2RowSize];
            for (int i = 0; i < mat2.GetLength(0); i++)
            {
                for (int j = 0; j < mat2RowSize; j++)
                {
                    for (int k = 0; k < mat1ColumnSize; k++)
                    {
                        output[k, j] += mat1[k, i] * mat2[i, j];
                    }
                }
            }

            return output;
        }

        public static float[,] TransposeMatrix(float[,] mat)
        {
            var transposedMat = new float[mat.GetLength(1), mat.GetLength(0)];
            for (int i = 0; i < mat.GetLength(0); i++)
            {
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    transposedMat[j, i] = mat[i, j];
                }
            }

            return transposedMat;
        }

        public static float RandomGaussian(float minValue = 0.0f, float maxValue = 1.0f)
        {
            float u;
            float s;

            do
            {
                u = 2.0f * Random.value - 1.0f;
                var v = 2.0f * Random.value - 1.0f;
                s = u * u + v * v;
            } while (s >= 1.0f);

            var std = u * Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);

            var mean = (minValue + maxValue) / 2.0f;
            var sigma = (maxValue - mean) / 3.0f;

            return Clamp(std * sigma + mean, minValue, maxValue);
        }

        public static float StandardDivination(float[,] values, float valuesMean = float.MinValue)
        {
            if (valuesMean == float.MinValue)
            {
                valuesMean = MatrixMean(values);
            }

            var valuesRowSize = values.GetLength(1);
            float sum = 0;
            for (int i = 0; i < values.GetLength(0); i++)
            {
                for (int j = 0; j < valuesRowSize; j++)
                {
                    var adjustedValue = values[i, j] - valuesMean;
                    sum += adjustedValue * adjustedValue;
                }
            }

            return Mathf.Sqrt(sum / values.Length);
        }

        public static float MatrixMean(float[,] mat)
        {
            var matRowSize = mat.GetLength(1);
            float result = 0;
            for (int i = 0; i < mat.GetLength(0); i++)
            {
                for (int j = 0; j < matRowSize; j++)
                {
                    result += mat[i, j];
                }
            }

            return result / mat.Length;
        }
    }
}