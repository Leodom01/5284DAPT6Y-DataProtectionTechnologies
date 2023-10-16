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
    def __init__(self, eventLog, k_saviour=True):
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
        self.__synthEnrichmentThreshold = 0.5
        self.__kSaviourEnabled = k_saviour
        self.__synthDataIncreaseBoundaries = (1.1, 1.5)

        self.__deletedMergedLogs = 0
        self.__addedLogs = 0
        self.__deletedMergedCases = 0
        self.__addedCases = 0

        self.__accurateAnnotationGeneration = True


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

    def getKvalues(self):
        # Return min, avg and max k anonimity
        k_vals = []

        for node in PreOrderIter(self._tree):
            if node != self._tree:
                k_vals.append(len(node.cases))

        k_vals = np.asarray(k_vals)

        return np.min(k_vals), np.mean(k_vals), np.max(k_vals)

    def getAlteredLogs(self):
        return self.__addedLogs, self.__deletedMergedLogs

    def getAlteredCases(self):
        return self.__addedCases, self.__deletedMergedCases

    def getNumberOfDifferentSequences(self):
        sequencesSet = set()
        for node in PreOrderIter(self._tree):
            sequencesSet.add(node.sequence)
        return len(sequencesSet)

    def __addAnnotation(self, annotation, activity):
        dataForActivity = self.__annotationDataOverAll.get(activity, None)
        if dataForActivity is None:
            self.__annotationDataOverAll[activity] = []
            dataForActivity = self.__annotationDataOverAll[activity]
        dataForActivity.append(annotation)

    def __setMaxDifferences(self):
        self.annotationMaxDifferences = dict()
        for key in self.__annotationDataOverAll.keys():
            maxVal = max(self.__annotationDataOverAll[key])
            minVal = min(self.__annotationDataOverAll[key])
            self.annotationMaxDifferences[key] = abs(maxVal - minVal)

    # node.name, node.annotations, t, node.cases
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

                    node_to_attach_to, generated_nodes = self._generate_nodes(node, nodes_to_add)
                    for currentNode in generated_nodes:
                        self._addNodeToTree(currentNode, node_to_attach_to)
                if len(node.cases) < k or self._violatesTCloseness(node.name, node.annotations, t, node.cases):
                    self.__deletedMergedCases += len(node.cases)
                    cutOutTraces = cutOutTraces.union(node.cases)
                    self._cutCasesOutOfTreeStartingFromNode(node, cutOutTraces)

                    if self._sequentialPrunning:
                        return cutOutTraces
        return cutOutTraces

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

    def _getAllPotentialSequencesTree(self, tree):
        return tree.sequences

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

    def getEventsOfNode(self, node):
        events = []
        if node != self._tree:
            events = events + [self.getEvent(case, node) for case in node.cases]
        return events

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

    def __generateDistanceMatrixSequences(self, sequences):
        distanceMatrix = dict()
        for sequence1 in sequences:
            distanceMatrix[sequence1] = dict()
            for sequence2 in sequences:
                if sequence1 != sequence2:
                    distanceMatrix[sequence1][sequence2] = levenshtein(sequence1, sequence2)
        return distanceMatrix

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

    def __areAllValuesInDistributionAreTheSame(self, distribution):
        if max(distribution) == min(distribution):
            return True
        else:
            return False

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

    def _getBucketLimits(self, t, overallDistribution):
        numberOfBuckets = round(t + 1)
        overallDistribution.sort()
        divider = round(len(overallDistribution) / numberOfBuckets)
        upperLimitsBuckets = list()
        for i in range(1, numberOfBuckets):
            upperLimitsBuckets.append(overallDistribution[min(round(i * divider), len(overallDistribution) - 1)])
        return upperLimitsBuckets

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

    def _generate_nodes(self, current_node, number_of_nodes):
        # Generates in the format of a list of lists, every inner list is: [activity, case, duration, event]

        # Use recursion to always start from the leaf of the branch
        if not current_node.is_leaf:
            child = current_node.children[0]
            return self._generate_nodes(child, number_of_nodes)
        else:
            # The new activity must obviously have the same activities of the branch we want to enrich
            activities = current_node.sequence.split("@")[1:]

            # Generate new case number
            all_cases_numbers = self._caseToSequenceDict.keys()
            start_case_num = int(sorted([int(key.split('-')[1]) for key in all_cases_numbers], reverse=True)[0])+1

        new_nodes = []
        for node_in_nodelist in range(number_of_nodes):
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

    def _value_from_specific_distribution(self, activity, sequence):
        # Sequence is a list of activities, to make a wiser choice in getting an accurate annotation
        # I could take more than 2 to make the prediction more accurate but for the sake of simplicity I'll take two
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
        last_two_weight = 0.7

        if np.random.uniform() <= last_two_weight:
            return np.random.normal(mean_last_two, std_last_two)
        else:
            return np.random.normal(mean_overall, std_overall)