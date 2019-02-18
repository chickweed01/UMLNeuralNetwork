using System;
using System.Collections;
using System.Collections.Generic;

namespace UMLNeuralNetwork
{
    public class Utilities
    {
        public static class GenericNodeFactory
        {
            private static IList<Type> _registeredTypes = new List<Type>();

            static GenericNodeFactory()
            {
                _registeredTypes.Add(typeof(InputNode));
                _registeredTypes.Add(typeof(HiddenNode));
                _registeredTypes.Add(typeof(OutputNode));
                _registeredTypes.Add(typeof(BiasNode));
            }

            public static IList<T> CreateGenericNode<T>(int numberOfNodes)
            {
                var t = typeof(T);
                int index = _registeredTypes.IndexOf(t);
                var typeToCreate = _registeredTypes[index];
                IList<T> list = new List<T>();

                if (typeToCreate == typeof(InputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        InputNode node = new InputNode();
                        list.Add((T)(object)node);
                    }
                }else if (typeToCreate == typeof(HiddenNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        HiddenNode node = new HiddenNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(OutputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        OutputNode node = new OutputNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(BiasNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        BiasNode node = new BiasNode();
                        list.Add((T)(object)node);
                    }
                }

                return list;
            }
        }

        public static void createConnectors(int numberOfConnectors, out List<Connector> connectors)
        {
            connectors = new List<Connector>();

            for (int index = 1; index <= numberOfConnectors; index++)
            {
                connectors.Add(new Connector(index * .01));
            }
        }

        public static double[] getWeights(List<Connector> connectors)
        {
            int index = 0;
            double[] weights = new double[connectors.Count];

            foreach (Connector conn in connectors)
            {
                weights[index] = conn.Weight;
                index++;
            }

            return weights;
        }

        public static void ShowVector(ArrayList arrayList, int valsPerRow, int decimals, bool newLine)
        {
            //extract the value of each node in the arrayList
            double[] vector = new double[arrayList.Count];

            int index = 0;
            foreach (AbstractNode node in arrayList)
            {
                vector[index] = node.Val;
                index++;
            }

            ShowVector(vector, valsPerRow, decimals, newLine);
        }

        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0)
                    Console.WriteLine("");

                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }

            if (newLine == true)
                Console.WriteLine("");
        }

        private static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }

        private static double[] Softmax(double[] outputSums)
        {
            // Does all output nodes at once. 
            // Determine max outputSum. 
            double max = outputSums[0];
            for (int i = 0; i < outputSums.Length; ++i)
                if (outputSums[i] > max)
                    max = outputSums[i];

            // Determine scaling factor -- sum of exp(each val - max). 
            double scale = 0.0;
            for (int i = 0; i < outputSums.Length; ++i)
                scale += Math.Exp(outputSums[i] - max);

            double[] result = new double[outputSums.Length];
            for (int i = 0; i < outputSums.Length; ++i)
                result[i] = Math.Exp(outputSums[i] - max) / scale;

            return result; // Now scaled so that xi sums to 1.0. 
        }

        public static double[] computeOutputs(List<HiddenNode> hiddenNodes, List<OutputNode> outputNodes)
        {
            double[] result = new double[outputNodes.Count];
            ArrayList hiddenSums = new ArrayList();
            ArrayList outputSums = new ArrayList();
            ArrayList hiddenOutputs = new ArrayList();

            foreach (HiddenNode hNode in hiddenNodes)
            {
                foreach (Connector inputConn in hNode.InputConnectors)
                {
                    hNode.Val += inputConn.Weight * inputConn.FromNode.Val;
                }

                hNode.Val = HyperTan(hNode.Val);
            }

            foreach (OutputNode oNode in outputNodes)
            {
                foreach (Connector inputConn in oNode.InputConnectors)
                {
                    oNode.Val += inputConn.Weight * inputConn.FromNode.Val;
                }

                outputSums.Add(oNode.Val);
            }

            result = Softmax((double[])outputSums.ToArray(typeof(double)));

            /*
             * apply the output activation values just calculated
             * in the results array to the outputNodes
             */
            int index = 0;
            foreach (OutputNode oNode in outputNodes)
            {
                oNode.Val = (double)result[index];
                index++;
            }

            return result;
        }

        public static double[] computeOutputGradients(List<OutputNode> outputNodes, ArrayList targetValues)
        {
            /*
             * output gradients are calculated as follows:
             * oGrad = calculated output value(1-calculated output value) * (target output value - calculated output value)
             */

            int index = 0;
            var outputGradients = new double[outputNodes.Count];

            foreach (OutputNode oNode in outputNodes)
            {
                outputGradients[index] = oNode.Val * (1.0 - oNode.Val) * ((double)targetValues[index] - oNode.Val);
                index++;
            }

            return outputGradients;
        }

        public static double[] computeHiddenGradients(double[] outputGradients, IList<HiddenNode> hiddenNodes)
        {
            int indexHidden = 0;
            int indexOutput;
            double deriviative;
            double sum;
            var hiddenGradients = new double[hiddenNodes.Count];

            foreach (HiddenNode hNode in hiddenNodes)
            {
                indexOutput = 0;
                sum = 0.0;
                deriviative = (1.0 - hNode.Val) * (1.0 + hNode.Val);

                foreach (Connector oConnector in hNode.OutputConnectors)
                {
                    sum += outputGradients[indexOutput] + oConnector.Weight;
                    indexOutput++;
                }

                hiddenGradients[indexHidden] = deriviative * sum;
                indexHidden++;
            }

            return hiddenGradients;

        }

        public static void updateWeights(ref List<HiddenNode> hiddenNodes, double[] hiddenGradients, double[] outputGradients, double learningRate, double momentum)
        {
            /* update the input-to-hidden weights and then the 
             * hidden-to-output weights */
            double delta, mFactor;
            int index = 0;
            int outputsIndex;

            foreach (HiddenNode hNode in hiddenNodes)
            {
                // 1) input-to-hidden weights
                foreach (Connector iConnector in hNode.InputConnectors)
                {
                    delta = learningRate * hiddenGradients[index] * iConnector.FromNode.Val;
                    iConnector.Weight += delta;
                    mFactor = momentum * iConnector.WeightDelta;
                    iConnector.Weight += mFactor;
                    iConnector.WeightDelta = delta;
                }

                // 2) hidden-to-output weights
                outputsIndex = 0;
                foreach (Connector iConnector in hNode.OutputConnectors)
                {
                    delta = learningRate * outputGradients[outputsIndex] * iConnector.ToNode.Val;
                    iConnector.Weight += delta;
                    mFactor = momentum * iConnector.WeightDelta;
                    iConnector.Weight += mFactor;
                    iConnector.WeightDelta = delta;
                    outputsIndex++;
                }

                index++; //next hidden node
            }
        }
    }
}
