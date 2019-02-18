using System;
using System.Collections;
using System.Collections.Generic;

namespace UMLNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin UML-based Artificial Neural Network demo\n");
            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;
            int numBiasNodes = numHidden + numOutput;
            int numberOfConnectors = (numInput * numHidden) + (numHidden * numOutput) + numBiasNodes;
            int epoch = 0;
            int maxEpochs = 600;
            var learningRate = 0.05;
            var momentum = 0.01;
            int iConnectorCounter = 0;
            int numHiddenBias = numHidden;
            int numOutputBias = numOutput;
            double[] xValues = new double[] { 1.0, 2.0, 3.0 };
            double[] outputs = new double[numOutput];
            
            List<InputNode> inputNodes;  List<HiddenNode> hiddenNodes; List<OutputNode> outputNodes; 
            List<BiasNode> biasNodes; List<Connector> connectors;
            ArrayList targetValues; 

            //specify target output values
            targetValues = new ArrayList(numOutput);
            targetValues.Add(.25);
            targetValues.Add(.75);

            Console.WriteLine("Creating a {0}-{1}-{2} tanh-softmax neural network", numInput, numHidden, numOutput);

            // create input nodes
            inputNodes = Utilities.GenericNodeFactory.CreateGenericNode<InputNode>(numInput) as List<InputNode>;
            //assign a value to each input node
            inputNodes[0].Val = 1.0;
            inputNodes[1].Val = 2.0;
            inputNodes[2].Val = 3.0;

            // create hidden nodes
            hiddenNodes = Utilities.GenericNodeFactory.CreateGenericNode<HiddenNode>(numHidden) as List<HiddenNode>;

            // create output node
            outputNodes = Utilities.GenericNodeFactory.CreateGenericNode<OutputNode>(numOutput) as List<OutputNode>;

            //create bias nodes            
            biasNodes = Utilities.GenericNodeFactory.CreateGenericNode<BiasNode>(numBiasNodes) as List<BiasNode>;

            /* create 26 connector nodes to hook up the 
             * input to hidden and hidden to output nodes and the bias nodes. */
            Utilities.createConnectors(numberOfConnectors, out connectors);

            // input nodes only have an output connector list
            foreach (InputNode iNode in inputNodes)
            {
               iNode.OutputConnectors = new List<Connector>();
            }

            //hidden nodes have both input and output connector lists
            foreach (HiddenNode hNode in hiddenNodes)
            {
                hNode.InputConnectors = new List<Connector>();
                hNode.OutputConnectors = new List<Connector>();
            }

            //output nodes have only an input connector list
            foreach (OutputNode oNode in outputNodes)
            {
                oNode.InputConnectors = new List<Connector>();
            }

            //set connectors for Bias nodes
            foreach (BiasNode bNode in biasNodes)
            {
                bNode.OutputConnectors = new List<Connector>();
            }

            //hook-up the connectors
            //input to hidden layer connections
            for (int i = 0; i < numInput; i++)
            {
                for (int j = 0; j < numHidden; j++)
                {
                    inputNodes[i].OutputConnectors.Add((Connector)connectors[iConnectorCounter]);
                    ((Connector)connectors[iConnectorCounter]).FromNode = inputNodes[i];
                    hiddenNodes[j].InputConnectors.Add((Connector)connectors[iConnectorCounter]);
                    ((Connector)connectors[iConnectorCounter]).ToNode = hiddenNodes[j];
                    iConnectorCounter++;
                }
            }

            //hidden to output layer connections
            for (int i = 0; i < numHidden; i++)
            {
                for (int j = 0; j < numOutput; j++)
                {
                    hiddenNodes[i].OutputConnectors.Add((Connector)connectors[iConnectorCounter]);
                    ((Connector)connectors[iConnectorCounter]).FromNode = hiddenNodes[i];
                    outputNodes[j].InputConnectors.Add((Connector)connectors[iConnectorCounter]);
                    ((Connector)connectors[iConnectorCounter]).ToNode = outputNodes[j];
                    iConnectorCounter++;
                }
            }

            //bias to hidden layer connections
            for (int i = 0; i < numHiddenBias; i++)
            {
                biasNodes[i].OutputConnectors.Add((Connector)connectors[iConnectorCounter]);
                ((Connector)connectors[iConnectorCounter]).FromNode = biasNodes[i];
                hiddenNodes[i].InputConnectors.Add((Connector)connectors[iConnectorCounter]);
                ((Connector)connectors[iConnectorCounter]).ToNode = hiddenNodes[i];
                iConnectorCounter++;
            }

            //add the bias nodes to the output layer nodes
            for (int i = 0; i < numOutputBias; i++)
            {
                biasNodes[i].OutputConnectors.Add((Connector)connectors[iConnectorCounter]);
                ((Connector)connectors[iConnectorCounter]).FromNode = biasNodes[i];
                outputNodes[i].InputConnectors.Add((Connector)connectors[iConnectorCounter]);
                ((Connector)connectors[iConnectorCounter]).ToNode = outputNodes[i];
                iConnectorCounter++;
            }

            Console.WriteLine("\nInputs are:");
            Utilities.ShowVector(xValues, 3, 1, true);

            Console.WriteLine("\nSetting default weights and biases:");
            Utilities.ShowVector(Utilities.getWeights(connectors), 8, 2, true);

            /*** loop until maxEpochs is reached ***/
            while (epoch <= maxEpochs)
            {
                // Compute and display outputs. 
                outputs = Utilities.computeOutputs(hiddenNodes, outputNodes);

                //compute gradients of output nodes
                var outputGradients = Utilities.computeOutputGradients(outputNodes, targetValues);

                //compute gradients of hidden nodes
                var hiddenGradients = Utilities.computeHiddenGradients(outputGradients, hiddenNodes);

                //update all weights and bias values
                Utilities.updateWeights(ref hiddenNodes, hiddenGradients, outputGradients, learningRate, momentum);

                if (epoch % 100 == 0)
                {
                    Console.WriteLine("\nEpoch = " + epoch.ToString() + " curr outputs = ");
                    Utilities.ShowVector(outputs, 2, 4, true);
                }

                ++epoch;
            }// *** end loop ***
            
            // final weights           
            Console.WriteLine("\nFinal weights and biases:");
            Utilities.ShowVector(Utilities.getWeights(connectors), 8, 2, true);

            Console.WriteLine("\nThe model outputs are:");
            Utilities.ShowVector(outputs, 2, 4, true);

            Console.WriteLine("\nEnd UML-based Artificial Neural Network demo\n");
            Console.ReadLine();
        }
    }
}
