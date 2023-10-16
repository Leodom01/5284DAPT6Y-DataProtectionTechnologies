from collections import Counter

from anytree import AnyNode, PreOrderIter, findall
from levenshtein import levenshtein
import sys
from scipy.stats import wasserstein_distance
from scipy.stats import normaltest
import pandas as pd
import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt


class Pretsa:
    def __init__(self, eventLog, k_saviour=True, enhanchedAnnGeneration=True):
        root = AnyNode(id='Root', name="Root", cases=set(), sequence="", annotation=dict(), sequences=set())
        current = root
        currentCase = ""
        caseToSequenceDict = dict()
        sequence = None
        self.__caseIDColName = "Case ID"
        self.__activityColName = "Activity"
        self.__annotationColName = "Duration"
        self.__constantEventNr = "Event_Nr"
        self.__annotationDataOverAll = dict()
        self.__normaltest_alpha = 0.05
        self.__normaltest_result_storage = dict()
        self.__normalTCloseness = True

        # K-pruning saviour settings
        # If a node has an anonymity set size which is more than 50% of what's required then we enrich the node logs
        # so to raise the anonymity set size and avoid losing or modifying the data.
        self.__synthEnrichmentThreshold = 0.5
        # The feature can be quickly enabled or disabled.
        self.__kSaviourEnabled = k_saviour
        # This defines how many new cases (made up by logs) we should create. 1 means that we create just the logs
        # necessary to overcome the k-anonymity set size threshold, 1.5 means that we want 50% more of whta's needed
        # and so on. It's a range because it will be a random value in order to make it harder to recognize what
        # are the nodes containing synthetic data.
        self.__synthDataIncreaseBoundaries = (1.1, 1.5)

        # Fields use to retrieve logging data for the graphs in the demo
        self.__deletedMergedLogs = 0
        self.__addedLogs = 0
        self.__deletedMergedCases = 0
        self.__addedCases = 0

        # The enhanced annotation generation can be quickly enabled or disabled.
        self.__accurateAnnotationGeneration = enhanchedAnnGeneration


        for index, row in eventLog.iterrows():
            activity = row[self.__activityColName]
            annotation = row[self.__annotationColName]
            if row[self.__caseIDColName] != currentCase:
                current = root
                if not sequence is None:
                    caseToSequenceDict[currentCase] = sequence
                    current.sequences.add(sequence)
                currentCase = row[self.__caseIDColName]
                current.cases.add(currentCase)
                sequence = ""
            childAlreadyExists = False
            sequence = sequence + "@" + activity
            for child in current.children:
                if child.name == activity:
                    childAlreadyExists = True
                    current = child
            if not childAlreadyExists:
                node = AnyNode(id=index, name=activity, parent=current, cases=set(), sequence=sequence,
                               annotations=dict())
                current = node
            current.cases.add(currentCase)
            current.annotations[currentCase] = annotation
            self.__addAnnotation(annotation, activity)
        # Handle last case
        caseToSequenceDict[currentCase] = sequence
        root.sequences.add(sequence)
        self._tree = root
        self._caseToSequenceDict = caseToSequenceDict
        self.__numberOfTracesOriginal = len(self._tree.cases)
        self._sequentialPrunning = True
        self.__setMaxDifferences()
        self.__haveAllValuesInActivitityDistributionTheSameValue = dict()
        self._distanceMatrix = self.__generateDistanceMatrixSequences(self._getAllPotentialSequencesTree(self._tree))

    """
        Calculate the minimum, average, and maximum anonymity set size of the whole tree.

        The minimum anonymity set size is also the k-anonymity of the dataset.
        Only returning k-anonymity would made hard to interpret since a set with only one entry
        could bring the whole dataset's k-anonymity down to 1, even if the rest of the dataset has high anonymity values.
        K-anonymity represents the minimum number of identical cases (logs) within a node.
        The function computes the minimum, average, and maximum k-anonymity values across all nodes in the tree.
        
        The anonimity set size value is the same as the number of cases in each node, which represents the number of 
        different logs with the same quasi identifier (it's the same metric used by pretsa to calculate the k-anonimity).

        Returns:
            Tuple (min_k, avg_k, max_k):
            - min_k (int): The minimum anonymity set size value observed in the tree (actual k anonimity).
            - avg_k (float): The average anonymity set size across all nodes.
            - max_k (int): The maximum anonymity set size observed in the tree.

        """
    def getKvalues(self):
        k_vals = []

        for node in PreOrderIter(self._tree):
            if node != self._tree:
                k_vals.append(len(node.cases))

        k_vals = np.asarray(k_vals)

        return np.min(k_vals), np.mean(k_vals), np.max(k_vals)

    """
        Get information about altered logs in the dataset.
        
        Confirmation of receipt-complete;case-10011;0.0;1
        T02 Check confirmation of receipt-complete;case-10011;67245.122;2
        Confirmation of receipt-complete;case-10017;0.0;1
        T06 Determine necessity of stop advice-complete;case-10017;27.271;2
        T10 Determine necessity to stop indication-complete;case-10017;0.0;3
        
        Every line of the dataset example is a log, while all the logs belonging to the same case ID made up a case.

        Returns:
            Tuple (added_logs, deleted_merged_logs):
            - added_logs (list): List of logs that were added.
            - deleted_merged_logs (list): List of logs that were deleted or merged with others, the action that 
                happens when for instance the k-anonymity or t-closeness isn't respected.
                
        """
    def getAlteredLogs(self):
        return self.__addedLogs, self.__deletedMergedLogs

    """
        Get information about altered cases in the dataset.

        Returns:
            Tuple (added_cases, deleted_merged_cases):
            - added_cases (list): List of cases that were added.
            - deleted_merged_cases (list): List of cases that were deleted or merged with others, the action that 
                happens when for instance the k-anonymity or t-closeness isn't respected.
            
        """
    def getAlteredCases(self):
        return self.__addedCases, self.__deletedMergedCases

    """
        Get the number of distinct sequences in the tree.
        
        This method calculates the number of distinct sequences within the tree, providing insights into the
        diversity of cases and their action sequences. The value will decrease during the sanitization process,
        indicating how much the variety of events is affected. Maintaining a high number of distinct sequences
        is essential for preserving an accurate representation of reality.

        Returns:
            int: The count of unique sequences in the tree.

        Example:
            Sequences: ["@A@B@C", "@A@B@B", "@B@A@C", "@A@B@C"]
                            ^         ^         ^         
            The number of sequences is 3, the last sequence is not added since it's the same as the first.    
            
        """
    def getNumberOfDifferentSequences(self):
        sequencesSet = set()
        for node in PreOrderIter(self._tree):
            sequencesSet.add(node.sequence)
        return len(sequencesSet)

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def __addAnnotation(self, annotation, activity):
        dataForActivity = self.__annotationDataOverAll.get(activity, None)
        if dataForActivity is None:
            self.__annotationDataOverAll[activity] = []
            dataForActivity = self.__annotationDataOverAll[activity]
        dataForActivity.append(annotation)

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def __setMaxDifferences(self):
        self.annotationMaxDifferences = dict()
        for key in self.__annotationDataOverAll.keys():
            maxVal = max(self.__annotationDataOverAll[key])
            minVal = min(self.__annotationDataOverAll[key])
            self.annotationMaxDifferences[key] = abs(maxVal - minVal)

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _violatesTCloseness(self, activity, annotations, t, cases):
        distributionActivity = self.__annotationDataOverAll[activity]
        maxDifference = self.annotationMaxDifferences[activity]
        # Consider only data from cases still in node
        distributionEquivalenceClass = []
        casesInClass = cases.intersection(set(annotations.keys()))
        for caseInClass in casesInClass:
            distributionEquivalenceClass.append(annotations[caseInClass])
        if len(distributionEquivalenceClass) == 0:  # No original annotation is left in the node
            return False
        if maxDifference == 0.0:  # All annotations have the same value(most likely= 0.0)
            return
        distances = []
        if self.__normalTCloseness == True:
            wasserstein_dist = wasserstein_distance(distributionActivity, distributionEquivalenceClass) / maxDifference
            distances.append(wasserstein_dist)
        else:
            return self._violatesStochasticTCloseness(distributionActivity, distributionEquivalenceClass, t, activity)

        return any(value > t for value in distances)

    """
        Prune the tree structure to satisfy k-anonymity and t-closeness contraint as explained in the paper.
        This function has been modified to contain my feature implementation: K-Saviour.

        K-Saviour avoid pruning the tree if a node doesn't satisfy the k-anonymity for a little margin.
        In that case a random (but sufficient to avoid pruning) number of new cases is generate and injected into the
        tree to avoid pruning the tree. 
        This allows us not to waste any precious data, not to lose whole sequences, and letting us seeing the whole
        picture describe by the highest number of distinct sequence possible. 
        This is particularly important when the event logs are analyzed with DFG (Direct Follows Graphs) as state in 
        the paper, it's important since DGF works on the sequences of events in a certain order, and it's important
        not to waste any stream of process.
        
        The K Saviour is only applied only when the t-closeness constraint is satisfied, if it is not satisfy the effort
        to "save" the equivalence class would be consistent and we would risk applying too relevant changes to the 
        original data.
    
        Args:
            k (int): The minimum k-anonymity value that every node must satisfy.
            t (float): The maximum t-closeness value that each node annotation must satisfy.

    """
    def _treePrunning(self, k, t):
        cutOutTraces = set()
        for node in PreOrderIter(self._tree):
            if node != self._tree:
                node.cases = node.cases.difference(cutOutTraces)
                if (self.__kSaviourEnabled and
                        k > len(node.cases) > self.__synthEnrichmentThreshold * k and
                        not self._violatesTCloseness(node.name, node.annotations, t, node.cases)):
                    # Then we have to enrich the data to have more than k entries in this equivalence class
                    nodes_to_add = math.ceil((k - len(node.cases)) * rnd.uniform(*self.__synthDataIncreaseBoundaries))
                    self.__addedCases += nodes_to_add

                    node_to_attach_to, generated_nodes = self._generate_cases(node, nodes_to_add)
                    for currentNode in generated_nodes:
                        self._addNodeToTree(currentNode, node_to_attach_to)
                if len(node.cases) < k or self._violatesTCloseness(node.name, node.annotations, t, node.cases):
                    self.__deletedMergedCases += len(node.cases)
                    cutOutTraces = cutOutTraces.union(node.cases)
                    self._cutCasesOutOfTreeStartingFromNode(node, cutOutTraces)

                    if self._sequentialPrunning:
                        return cutOutTraces
        return cutOutTraces

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _cutCasesOutOfTreeStartingFromNode(self, node, cutOutTraces, tree=None):
        if tree == None:
            tree = self._tree
        current = node
        try:
            tree.sequences.remove(node.sequence)
        except KeyError:
            pass
        while current != tree:
            current.cases = current.cases.difference(cutOutTraces)
            self.__deletedMergedLogs += len(cutOutTraces)
            if len(current.cases) == 0:
                node = current
                current = current.parent
                node.parent = None
            else:
                current = current.parent

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _getAllPotentialSequencesTree(self, tree):
        return tree.sequences

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _addCaseToTree(self, trace, sequence, tree=None):
        if tree == None:
            tree = self._tree
        if trace != "":
            activities = sequence.split("@")
            currentNode = tree
            tree.cases.add(trace)
            for activity in activities:
                for child in currentNode.children:
                    if child.name == activity:
                        child.cases.add(trace)
                        currentNode = child
                        break

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def __combineTracesAndTree(self, traces):
        # We transform the set of sequences into a list and sort it, to discretize the behaviour of the algorithm
        sequencesTree = list(self._getAllPotentialSequencesTree(self._tree))
        sequencesTree.sort()
        for trace in traces:
            bestSequence = ""
            # initial value as high as possible
            lowestDistance = sys.maxsize
            traceSequence = self._caseToSequenceDict[trace]
            for treeSequence in sequencesTree:
                currentDistance = self._getDistanceSequences(traceSequence, treeSequence)
                if currentDistance < lowestDistance:
                    bestSequence = treeSequence
                    lowestDistance = currentDistance
            self._overallLogDistance += lowestDistance
            self._addCaseToTree(trace, bestSequence)

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def runPretsa(self, k, t, normalTCloseness=True):
        self.__normalTCloseness = normalTCloseness
        if not self.__normalTCloseness:
            self.__haveAllValuesInActivitityDistributionTheSameValue = dict()
        self._overallLogDistance = 0.0
        if self._sequentialPrunning:
            cutOutCases = set()
            cutOutCase = self._treePrunning(k, t)
            while len(cutOutCase) > 0:
                self.__combineTracesAndTree(cutOutCase)
                cutOutCases = cutOutCases.union(cutOutCase)
                cutOutCase = self._treePrunning(k, t)
        else:
            cutOutCases = self._treePrunning(k, t)
            self.__combineTracesAndTree(cutOutCases)
        return cutOutCases, self._overallLogDistance

    """
        Generate new annotation based on the activity and the previous activity in the case log.
        If the __accurateAnnotationGeneration value is True the enhanced annotation generation feature is used.
        
        The pretsa original annotation generation (still present in the code) returned the annotation by sampling a 
        normal distribution. The normal distribution would have average value and standard deviation computed by 
        taking into account all the annotation values of all the logs with the same activity as the one in the 
        activity parameter.
        An example: 
            Confirmation of receipt-complete;case-10011;0.0
            T02 Check confirmation of receipt-complete;case-10011;28
            T10 Determine necessity to stop indication-complete;case-10011; <TO GENERATE>
            Confirmation of receipt-complete;case-10017;0.0
            T06 Determine necessity of stop advice-complete;case-10017;5
            T10 Determine necessity to stop indication-complete;case-10017;100
            T02 Determine necessity of stop advice-complete;case-10018;0
            T10 Determine necessity to stop indication-complete;case-10018;20
            
        The normal pretsa distribution would be created by taking all the annotations from the logs which have T10
        as an activity. In this case the T10 annotation values would be taken from case-10018 (annotation: 20) and
        from case-10017 (annotation: 100). 
        The problem is that the annotation (the duration) is time it takes to go from the action before T10 to T10, 
        so the T10 annotation (or duration) in this case is not strictly correlated to the action itself but to the 
        previous action.
        
        The enhanced annotation generation takes into deep consideration the activity that come previous to the activity 
        we want to generate the annotation for.
        The annotation is sampled from a bimodal distribution. The first distribution creating the bimodal distribution 
        is the one used in the standard annotation generation (all the logs having the same activity of the log to 
        generate). The second one (that has twice the weight of the first) is generate from logs beloning to similar 
        activity sequences, in particular the distribution is created by all the annotations from the logs that have
        the same activity and whose previous log have the same activity of the log we have to create the annotation for.
        According to the previous dataset: the first distribution would be created by all the logs with action T10 
        (in that case, the cases 10018 and 10017). The second distribution will be create by the log T10 but only the one
        belonging to case-10018 since in that case the sequence is T01->T10 just like in case-10011 which is the one 
        we want to generate the annotation for.
        
        Since the DFG algorithms used to exploit this data (as state in the paper) works on the correlations between 
        consequent events it is important to keep the annotation correlations between sequence of actions as close as 
        possible to the real data.
        
        In short:
         Action A ---> B ---> C 
         Action A ---> D ---> C
         These two cases cannot have the same relevance on defining the annotation for the C action logs.
    
        Args:
            activity (string): The activity that we have to add the annotation to.
            sequence [string]: The activity and the previous activity of the log we have to generate the annotation for.
    """
    def __generateNewAnnotation(self, activity, sequence):
        # normaltest works only with more than 8 samples
        if (
                len(self.__annotationDataOverAll[
                        activity])) >= 8 and activity not in self.__normaltest_result_storage.keys():
            stat, p = normaltest(self.__annotationDataOverAll[activity])
        else:
            p = 1.0
        self.__normaltest_result_storage[activity] = p
        # I realize it is quite expensive to do this calculation every time, a caching system with invalidation
        # could be implemented for big datasets
        if self.__accurateAnnotationGeneration and len(sequence) > 1:
            randomValue = self._value_from_specific_distribution(activity, sequence)
        elif self.__normaltest_result_storage[activity] <= self.__normaltest_alpha:
            mean = np.mean(self.__annotationDataOverAll[activity])
            std = np.std(self.__annotationDataOverAll[activity])
            randomValue = np.random.normal(mean, std)
        else:
            randomValue = np.random.choice(self.__annotationDataOverAll[activity])

        if randomValue < 0:
            randomValue = 0

        return randomValue

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def getEvent(self, case, node):
        event = {
            self.__activityColName: node.name,
            self.__caseIDColName: case,
            self.__annotationColName: node.annotations.get(case,
                    self.__generateNewAnnotation(node.name,
                                                 self._caseToSequenceDict[case].split("@")[1:node.depth+1])),
            self.__constantEventNr: node.depth
        }
        return event

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def getEventsOfNode(self, node):
        events = []
        if node != self._tree:
            events = events + [self.getEvent(case, node) for case in node.cases]
        return events

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def getPrivatisedEventLog(self):
        events = []
        self.__normaltest_result_storage = dict()
        nodeEvents = [self.getEventsOfNode(node) for node in PreOrderIter(self._tree)]
        for node in nodeEvents:
            events.extend(node)
        eventLog = pd.DataFrame(events)
        if not eventLog.empty:
            eventLog = eventLog.sort_values(by=[self.__caseIDColName, self.__constantEventNr])
        return eventLog

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def __generateDistanceMatrixSequences(self, sequences):
        distanceMatrix = dict()
        for sequence1 in sequences:
            distanceMatrix[sequence1] = dict()
            for sequence2 in sequences:
                if sequence1 != sequence2:
                    distanceMatrix[sequence1][sequence2] = levenshtein(sequence1, sequence2)
        return distanceMatrix

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _getDistanceSequences(self, sequence1, sequence2):
        if sequence1 == "" or sequence2 == "" or sequence1 == sequence2:
            return sys.maxsize
        try:
            distance = self._distanceMatrix[sequence1][sequence2]
        except KeyError:
            print("A Sequence is not in the distance matrix")
            print(sequence1)
            print(sequence2)
            raise
        return distance

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def __areAllValuesInDistributionAreTheSame(self, distribution):
        if max(distribution) == min(distribution):
            return True
        else:
            return False

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _violatesStochasticTCloseness(self, distributionEquivalenceClass, overallDistribution, t, activity):
        if activity not in self.__haveAllValuesInActivitityDistributionTheSameValue.keys():
            self.__haveAllValuesInActivitityDistributionTheSameValue[
                activity] = self.__areAllValuesInDistributionAreTheSame(overallDistribution)
        if not self.__haveAllValuesInActivitityDistributionTheSameValue[activity]:
            upperLimitsBuckets = self._getBucketLimits(t, overallDistribution)
            return (self._calculateStochasticTCloseness(overallDistribution, distributionEquivalenceClass,
                                                        upperLimitsBuckets) > t)
        else:
            return False

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _calculateStochasticTCloseness(self, overallDistribution, equivalenceClassDistribution, upperLimitBuckets):
        overallDistribution.sort()
        equivalenceClassDistribution.sort()
        counterOverallDistribution = 0
        counterEquivalenceClass = 0
        distances = list()
        for bucket in upperLimitBuckets:
            lastCounterOverallDistribution = counterOverallDistribution
            lastCounterEquivalenceClass = counterEquivalenceClass
            while counterOverallDistribution < len(overallDistribution) and overallDistribution[
                counterOverallDistribution
            ] < bucket:
                counterOverallDistribution = counterOverallDistribution + 1
            while counterEquivalenceClass < len(equivalenceClassDistribution) and equivalenceClassDistribution[
                counterEquivalenceClass
            ] < bucket:
                counterEquivalenceClass = counterEquivalenceClass + 1
            probabilityOfBucketInEQ = (counterEquivalenceClass - lastCounterEquivalenceClass) / len(
                equivalenceClassDistribution)
            probabilityOfBucketInOverallDistribution = (
                                                               counterOverallDistribution - lastCounterOverallDistribution) / len(
                overallDistribution)
            if probabilityOfBucketInEQ == 0 and probabilityOfBucketInOverallDistribution == 0:
                distances.append(0)
            elif probabilityOfBucketInOverallDistribution == 0 or probabilityOfBucketInEQ == 0:
                distances.append(sys.maxsize)
            else:
                distances.append(max(probabilityOfBucketInEQ / probabilityOfBucketInOverallDistribution,
                                     probabilityOfBucketInOverallDistribution / probabilityOfBucketInEQ))
        return max(distances)

    """"
        The feature has been left untouched from the original pretsa implementation.
    """
    def _getBucketLimits(self, t, overallDistribution):
        numberOfBuckets = round(t + 1)
        overallDistribution.sort()
        divider = round(len(overallDistribution) / numberOfBuckets)
        upperLimitsBuckets = list()
        for i in range(1, numberOfBuckets):
            upperLimitsBuckets.append(overallDistribution[min(round(i * divider), len(overallDistribution) - 1)])
        return upperLimitsBuckets

    """"
        Adds a new node to the tree, it is used when we want to add a new case to out dataset to follow the 
        k-anonymity constraint and to avoid pruning the tree for the upmentioned reasons.
    
        Args:
            node (anytree node) : The node that we want to add to the three.
            injectionPoint (anytree node): The node from which we want to add the case to.
    """
    def _addNodeToTree(self, node, injectionPoint):
        sequence = ""
        for entry in node:
            sequence = sequence + "@" + entry["activity"]
        for entry in node:
            self.__addAnnotation(entry["duration"], entry["activity"])

        # Updating pretsa level data structures
        self._addCaseToTree(node[0]["case"], sequence)
        self._caseToSequenceDict[node[0]["case"]] = sequence

        # Updating individual nodes
        currentNode = injectionPoint
        while currentNode.parent is not None:
            currentNode.cases.add(node[0]["case"])
            currentNode = currentNode.parent

    """"
        Generate a new case, which will then need to be added to the tree. The generated cases will be compatible with 
        the dataset in terms of case number and annotation values.
        
        The annotation (duration) values are generated through the __generateNewAnnotation method, the case number is always equals
        to the biggest case number+1.
        Every case contains a number of nodes, every node is one log to be added in the dataset.

        Args:
            current_node (anytree node) : The node that we want to enrich with new cases.
            number_of_cases (int): The number of cases that we want to add.
            
        Returns:
            anytree: The leaf node from which we need to start adding the returned cases (the method generates the case, 
                it doesn't add them to the tree)
            [[logs]]: a list of lists of logs, a list of logs describes a new case.
    """
    def _generate_cases(self, current_node, number_of_cases):
        # Generates in the format of a list of lists, every inner list is: [activity, case, duration, event]

        # Use recursion to always start from the leaf of the branch
        if not current_node.is_leaf:
            child = current_node.children[0]
            return self._generate_cases(child, number_of_cases)
        else:
            # The new activity must obviously have the same activities of the branch we want to enrich
            activities = current_node.sequence.split("@")[1:]

            # Generate new case number
            all_cases_numbers = self._caseToSequenceDict.keys()
            start_case_num = int(sorted([int(key.split('-')[1]) for key in all_cases_numbers], reverse=True)[0])+1

        new_nodes = []
        for node_in_nodelist in range(number_of_cases):
            new_node = []
            # Generate every "line" of the new activities to log
            for event_nr in range(len(activities)):
                activity_to_add = {"activity": activities[event_nr],
                                   "case": "case-" + str(start_case_num+node_in_nodelist),
                                   "duration": self.__generateNewAnnotation(activities[event_nr], activities[:event_nr+1]),
                                   "event_nr": event_nr}
                new_node.append(activity_to_add)
                self.__addedLogs += 1
            new_nodes.append(new_node)

        return current_node, new_nodes

    """"
        Returns a value sampled from the bimodal distribution described in the __generateNewAnnotation method.
        
        A bimodal distribution is made and a value sampled from that is added. In the future the method could 
        implement some caching or memoization techniques since it's pretty heavy in terms of computation.
        
        Args:
            activity (string): The activity that we have to add the annotation to.
            sequence [string]: The activity and the previous activity of the log we have to generate the annotation for.
        
        Returns:
            float: A value sampled from the bimodal distribution.
       """
    def _value_from_specific_distribution(self, activity, sequence):
        # Using bimodal distribution
        two_last_activities = sequence[-2:]
        # Cases that have a sequence containing consecutive two_last_activities
        cases_match_activities = set()
        for key in self._caseToSequenceDict.keys():
            value = self._caseToSequenceDict[key]
            if value.__contains__("@"+two_last_activities[0]+"@"+two_last_activities[1]):
                cases_match_activities.add(key)

        # All the annotations
        annotations_for_two_last = []
        # Nodes that have a sequence containing consecutive two_last_activities
        nodes_last_activities = findall(self._tree,
                                                     lambda node: node.cases.intersection(cases_match_activities) and
                                                            node.sequence.endswith("@"+two_last_activities[0]+"@"+two_last_activities[1])
                                        )
        for node in nodes_last_activities:
            annotations_for_two_last.extend([node.annotations[key] for key in node.annotations.keys()
                                             if key in cases_match_activities])

        # Here are the two distribution that will made up the bimodal distribution
        # First the normal standard distribution already used in pretsa
        mean_overall = np.mean(self.__annotationDataOverAll[activity])
        std_overall = np.std(self.__annotationDataOverAll[activity])
        # In case there are no nodes with sequence similar to mine I'll use the overall distribution as original Pretsa
        if len(nodes_last_activities) > 0:
            mean_last_two = np.mean(annotations_for_two_last)
            std_last_two = np.std(annotations_for_two_last)
        else:
            mean_last_two = mean_overall
            std_last_two = std_overall

        # I want to sample from the mean distribution between those two, but the last_two has a double weight
        last_two_weight = 0.67

        if np.random.uniform() <= last_two_weight:
            return np.random.normal(mean_last_two, std_last_two)
        else:
            return np.random.normal(mean_overall, std_overall)