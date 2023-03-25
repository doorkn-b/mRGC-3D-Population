#!/usr/bin/env python
############################################################################
#           generate_hoc.py - Automated NEURON HOC code generator.
#                              -------------------
#     begin           : Sat 12 Apr 2014 16:09:08 EST
#     copyright       : (C) 2014 by D Tsai, S Bai, T Guo
#     email           : dtsai@computer.org
#     ID              : $Id$
############################################################################

import math
import string
import re
import sys
import time
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from foveal_tiles import cell_type_to_diameter, get_ais_diameter

VERSION_STRING = "(V1.0  build Mon 28 Apr 2014 22:17:25 EDT)"


############################################################################
# Globals
lineCount = 0
procBlock = 0

# NOTE: Increase recursion limit needed for huge axons
#           - do NOT blindly change this value
sys.setrecursionlimit(2500)

############################################################################
# Tree classes


class Node:
    """Generic class for tree nodes."""

    def __init__(self, topoId, buildId, parentNode=None):
        self.topoId = topoId  # node topological ID
        self.buildId = buildId  # ID for building hoc code
        self.parent = parentNode  # ref parent node
        self.points = []  # nodal point list
        self.children = []  # children node list
        self.diameter = 1.0  # node diameter
        self.length = 0.0  # topological length

    def addPoint(self, x, y, z):
        self.points.append([x, y, z])
        if len(self.points) >= 2:
            p = self.points[-1]
            q = self.points[-2]
            self.length = self.length + math.sqrt(
                math.pow(q[0] - p[0], 2)
                + math.pow(q[1] - p[1], 2)
                + math.pow(q[2] - p[2], 2)
            )
            return self.length

    def len(self):
        return self.length

    def addChildNode(self, node):
        self.children.append(node)
        return node

    def __repr__(self):
        pointList = []
        for n in self.points:
            pointList.append(n)
        childList = []
        for n in self.children:
            childList.append(n.__repr__())
        return (
            str(self.topoId)
            + " "
            + str(self.buildId)
            + " "
            + str(pointList)
            + " "
            + string.replace(str(childList), "'", "")
        )


class Soma(Node):
    """Representation of somatic object."""

    def __repr__(self):
        return "(S " + Node.__repr__(self) + ")"

    def typeString(self):
        return "soma"


class Dendrite(Node):
    """Representation of dendritic objects."""

    def __repr__(self):
        return "(D " + Node.__repr__(self) + ")"

    def typeString(self):
        return "dend"


class Hillock(Node):
    """Representation of axon hillock objects."""

    def __repr__(self):
        return "(H " + Node.__repr__(self) + ")"

    def typeString(self):
        return "hillock"


class AIS(Node):
    """Representation of AIS objects."""

    def __repr__(self):
        return "(X " + Node.__repr__(self) + ")"

    def typeString(self):
        return "ais"


class Axon(Node):
    """Representation of axon objects."""

    def __repr__(self):
        return "(A " + Node.__repr__(self) + ")"

    def typeString(self):
        return "axon"


############################################################################
# Parser and tree builder


def buildTree(file):
    """
    Builds a cell from topological information contained in file pointed
    to by `file'. Returns the root node.
    """

    lineCount = 1
    rootNode = None
    nodeDB = {}

    somaCount = 0
    dendCount = 0
    hillockCount = 0
    aisCount = 0
    axonCount = 0

    cell_type = ""

    for line in file:
        line = line.strip()
        if line == "":
            continue
        if line[0] == "#":
            if "ON" in line:
                cell_type = "ON"
            elif "OFF" in line:
                cell_type = "OFF"
            continue
        lineCount += 1

        # parse fields
        line = re.sub("NaN", "-1", line)
        [type, x, y, z, xp, yp, zp, id, idparent, seglen] = line.split(" ")
        type = int(type)
        x = float(x)
        y = float(y)
        z = float(z)
        id = int(float(id))
        idparent = int(idparent)
        # print "| ", type, x, y, z, id, idparent

        # find parent node
        if idparent != -1:
            parentNode = nodeDB[idparent]
        else:
            parentNode = None

        # insert node
        if type == 0:
            node = Soma(id, somaCount, parentNode)
            node.addPoint(x, y, z)
            somaCount += 1
            if parentNode is not None:
                parentNode.addChildNode(node)
        elif type == 1:
            node = Dendrite(id, dendCount, parentNode)
            node.addPoint(x, y, z)
            dendCount += 1
            if parentNode is not None:
                parentNode.addChildNode(node)
        elif type == 2:
            node = Hillock(id, hillockCount, parentNode)
            node.addPoint(x, y, z)
            hillockCount += 1
            if parentNode is not None:
                parentNode.addChildNode(node)
        elif type == 3:
            node = AIS(id, aisCount, parentNode)
            node.addPoint(x, y, z)
            aisCount += 1
            if parentNode is not None:
                parentNode.addChildNode(node)
        elif type == 4:
            node = Axon(id, axonCount, parentNode)
            node.addPoint(x, y, z)
            axonCount += 1
            if parentNode is not None:
                parentNode.addChildNode(node)
        else:
            raise BaseException("line %d: invalid section type %s" % (lineCount, type))
        nodeDB[id] = node

        # sanity check
        if id != 0 and parentNode is None:
            raise BaseException(
                "line %d: only root node can be parentless" % (lineCount)
            )

        # climb tree
        if rootNode is None:
            rootNode = node

    return rootNode, cell_type


############################################################################
# Support procedures


def countNodeTypes(node):
    """
    Counts the number of Soma, Hillock, AIS, Axon, and Dendrite nodes, starting
    at `node' and recursively through all child nodes below it. Returns the
    values as a list.
    """

    somaNum = 0
    dendNum = 0
    hillockNum = 0
    aisNum = 0
    axonNum = 0

    if node.typeString() == "soma":
        somaNum += 1
    elif node.typeString() == "dend":
        dendNum += 1
    elif node.typeString() == "hillock":
        hillockNum += 1
    elif node.typeString() == "ais":
        aisNum += 1
    elif node.typeString() == "axon":
        axonNum += 1
    else:
        raise BaseException("unknown node class")

    for c in node.children:
        [s, d, h, x, a] = countNodeTypes(c)
        somaNum += s
        dendNum += d
        hillockNum += h
        aisNum += x
        axonNum += a

    return [somaNum, dendNum, hillockNum, aisNum, axonNum]


def bridgeDisjointNodes(node):
    """
    Go through entire tree below `node' and ensure each node physically
    contacts its immediate parent node, i.e. all parent-child pair share one
    common (x,y,z) location in the Node.points list.
    """

    for child in node.children:
        connected = 0
        for point in child.points:
            if point in node.points:
                connected = 1
                break
        if not connected:
            child.points.insert(0, node.points[-1])

        bridgeDisjointNodes(child)


def fixSingularPointSoma(node, somaDiameter):
    """
    Make sure soma node(s) is not a single point in space. If this is the case,
    an additional point is inserted, with the x coord value equalling the existing
    x value minus the somatic diameter `somaDiameter'.
    """

    if node.typeString() == "soma" and len(node.points) == 1:
        [x, y, z] = node.points[0]
        node.points.append([x + somaDiameter, y, z])

    for child in node.children:
        fixSingularPointSoma(child, somaDiameter)


def setNodeDiameter(node, distToRoot, dendDiam, somDiam, hillDiam, aisDiam, axDiam):
    """
    Sets the diameter of all tree nodes starting at `node'. `distToRoot' is the
    distance of `node' from the root node. The parameters `dendDiam', `somDiam',
    `hillDiam', 'aisDiam', and `axDiam' are the scalar somatic, hillock, AIS, and
    axonal diameter.
    """

    if node.typeString() == "soma":
        node.diameter = somDiam
    elif node.typeString() == "dend":
        node.diameter = dendDiam
    elif node.typeString() == "hillock":
        node.diameter = hillDiam
    elif node.typeString() == "ais":
        node.diameter = aisDiam
    elif node.typeString() == "axon":
        node.diameter = axDiam
    else:
        raise BaseException("unknown node class")

    for child in node.children:
        setNodeDiameter(
            child,
            distToRoot + node.length / 2.0,
            dendDiam,
            somDiam,
            hillDiam,
            aisDiam,
            axDiam,
        )


def continuousSegments(node):
    """
    Returns the last node in a continuous stretch of non-branching segments
    of identical type. Returns the current node if such condition is not met.
    """

    if len(node.children) != 1:
        return node

    c = node.children[0]
    if c.buildId != node.buildId + 1 or c.typeString() != c.parent.typeString():
        return node
    else:
        return continuousSegments(c)


def continuousBranching(node):
    """
    Look for branching dendritic point(s), where the child nodes have
    consecutive buildId. Returns the lowest and the highest ID number in a
    list, otherwise returns an empty list.
    """

    buildIds = [c.buildId for c in node.children if c.typeString() == "dend"]
    if len(buildIds) < 2:
        return []

    buildIds.sort
    first = 0
    last = 0
    for i in range(1, len(buildIds)):
        if buildIds[i] == buildIds[i - 1] + 1:
            last = i
        else:
            first = i
            last = i

    if first != last:
        return [buildIds[first], buildIds[last]]
    else:
        return []


############################################################################
# Code generators


def printPreamble(file):
    """Print code generator version, and creation date."""

    file.write(
        "// AUTO GENERATED CODE --- DO NOT MODIFY\n"
        "// Created by generate_hoc %s\n"
        "// On %s\n\n" % (VERSION_STRING, time.strftime("%d/%m/%Y"))
    )


def printHeaderBlock(modelName, tree, file):
    """
    Prints header block to file pointed to by `file' for the given syntax
    tree.
    """

    printPreamble(file)
    file.write("begintemplate %s\n\n" % (modelName))
    file.write("  public dend, soma, hillock, ais, axon\n")
    file.write("  public all, dendrites, hillocks, aises, axons\n")

    [s, d, h, x, a] = countNodeTypes(tree)
    file.write(
        "  create soma[%d], dend[%d], hillock[%d], ais[%d], axon[%d]\n\n"
        % (s, d, h, x, a)
    )
    file.write("  objref all, dendrites, hillocks, aises, axons\n\n")

    file.write(
        "  proc init() { local x, y, z\n"
        "    // for specifying positions of a cell network\n"
        "    if (numarg() == 3) {\n"
        "      x = $1\n"
        "      y = $2\n"
        "      z = $3\n"
        "    } else { \n"
        "      x = 0\n"
        "      y = 0\n"
        "      z = 0\n"
        "    }\n"
        "\n"
        "    // build cell\n"
        "    access soma\n"
        "    topol(x, y, z)\n"
        "    customizeTopol(x, y, z)\n"
        "    biophysics()\n"
        "  }\n\n"
    )

    file.write(
        "  proc customizeTopol() {\n"
        "    //section lists\n"
        "    all = new SectionList()\n"
        "    all.wholetree()\n"
        "    dendrites = new SectionList()\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      dend[i] dendrites.append()\n"
        "    }\n"
        "    hillocks = new SectionList()\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      hillock[i] hillocks.append()\n"
        "    }\n"
        "    aises = new SectionList()\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      ais[i] aises.append()\n"
        "    }\n"
        "    axons = new SectionList()\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      axon[i] axons.append()\n"
        "    }\n"
        "\n"
        "    /*// added axon convergence point\n"
        "    axon {\n"
        "      pt3dadd($1+800, $2+427, $3-4.85, 0.27)\n"
        "    }\n"
        "    define_shape()*/\n"
        "\n"
        "    forall {\n"
        "      // segments length < 12 um\n"
        "      if (L/nseg > 50) {  //XXX: increase to 50\n"
        "        nseg = int( (L/nseg) / 50 + 1 )\n"
        "      }\n"
        "      if (nseg %% 2 == 0) {\n"
        "        nseg += 1\n"
        "      }\n"
        "    }\n"
        "\n"
        "    // additional geometry for ais narrower axon\n"
        "    soma    { nseg >= 3 }\n"
        "    hillock { diam = 1.0 }\n"
        "    ais     { diam = 0.8 nseg >= 5}\n"
        "    axon    { diam = 1.0 }\n"
        "  }\n\n" % (d, h, x, a)
    )


def printBiophysBlock(tree, file):
    """
    Prints biophysics related code to file pointed to by `file' for the
    syntax `tree'.
    """

    file.write(
        "  proc biophysics() {\n"
        "    // all conductances are in S/cm^2\n"
        "    forall insert pas\n"
        "    forall insert spike\n"
        "    forall ena = 35.0\n"
        "    forall ek = -75\n"
        "    forall insert cad\n"
        "    forall g_pas = 0.000005\n"
        "    forall e_pas = -62.5\n"
        "    forall Ra=110\n\n"
    )

    ## SPIKE ##

    file.write(
        "    forsec dendrites {\n"
        "      gnabar_spike = 0.040\n"
        "      gkbar_spike  = 0.012\n"
        "      gabar_spike  = 0.036\n"
        "      gcabar_spike = 0.002\n"
        "      gkcbar_spike = 0.00005\n"
        "    }\n"
        "\n"
        "    soma {\n"
        "      gnabar_spike = 0.070\n"
        "      gkbar_spike  = 0.018\n"
        "      gabar_spike  = 0.054\n"
        "      gcabar_spike = 0.0015\n"
        "      gkcbar_spike = 0.000065\n"
        "    }\n"
        "\n"
    )

    [s, d, h, x, a] = countNodeTypes(tree)
    file.write(
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      hillock[i] {\n"
        "        gnabar_spike = 0.070\n"
        "        gkbar_spike  = 0.018\n"
        "        gabar_spike  = 0.054\n"
        "        gcabar_spike = 0.0015\n"
        "        gkcbar_spike = 0.000065\n"
        "      }\n"
        "    }\n"
        "\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      ais[i] {\n"
        "        gnabar_spike = 0.700  //10x of soma\n"
        "        gkbar_spike  = 0.018  //matched\n"
        "        gabar_spike  = 0.054\n"
        "        gcabar_spike = 0.0015\n"
        "        gkcbar_spike = 0.000065\n"
        "      }\n"
        "    }\n"
        "\n"
        "    for (i = 0; i < %d; i=i+1) {\n"
        "      axon[i] {\n"
        "        gnabar_spike = 0.070\n"
        "        gkbar_spike  = 0.018\n"
        "        gabar_spike  = 0\n"
        "        gcabar_spike = 0\n"
        "        gkcbar_spike = 0.000065\n"
        "      }\n"
        "    }\n"
        "\n" % (h, x, a)
    )

    file.write("    forall depth_cad = diam/2\n" "  }\n\n")


def printConnectBlockNodes(node, file):
    """Recursively generates topology related connection code for this node
    and its children."""

    global lineCount, procBlock

    # limit proc block length
    lineCount += 1
    if lineCount > 128:
        file.write("  }\n")
        file.write("  proc connect_%d() { local i\n" % (procBlock + 1))
        lineCount = 0
        procBlock += 1

    term = continuousSegments(node)
    twoPlusSegs = 0
    if term != node and node.children[0] != term:
        twoPlusSegs = 1
    branchIds = continuousBranching(node)

    if term != node and twoPlusSegs:
        # identical segment stretch
        file.write(
            "    for i = %d, %d connect %s[i](0), %s[i-1](1)\n"
            % (node.buildId + 1, term.buildId, node.typeString(), node.typeString())
        )
        printConnectBlockNodes(term, file)
    elif branchIds != []:
        # consecutive branches
        file.write(
            "    for i = %d, %d connect dend[i](0), dend[%d](1)\n"
            % (branchIds[0], branchIds[-1], node.buildId)
        )
        # non-consecutive branches
        for c in node.children:
            if c.buildId not in range(branchIds[0], branchIds[-1] + 1):
                file.write(
                    "    connect %s[%d](0), %s[%d](1)\n"
                    % (c.typeString(), c.buildId, node.typeString(), node.buildId)
                )
        for c in node.children:
            printConnectBlockNodes(c, file)
    else:
        # mixed types
        for c in node.children:
            file.write(
                "    connect %s[%d](0), %s[%d](1)\n"
                % (c.typeString(), c.buildId, node.typeString(), node.buildId)
            )
        for c in node.children:
            printConnectBlockNodes(c, file)


def printConnectBlock(tree, file):
    """
    Generates topology related "connect" code, for the given syntax tree, into
    the specified file.
    """

    # reset proc block length counter
    global lineCount, procBlock
    lineCount = 0
    procBlock = 0

    file.write("  proc connect_0() { local i\n")
    printConnectBlockNodes(tree, file)
    file.write("  }\n")


def printTopolBlock(tree, file):
    """Generate topology related code in `file'."""

    file.write("\n  proc topol () {\n")
    for i in range(procBlock + 1):
        file.write("    connect_%d($1, $2, $3)\n" % i)
    file.write("    basic_shape($1, $2, $3)\n" "  }\n\n")


def printShapeBlockNode(node, file):
    """Recursively generate shape related code, starting at `node', into `file'."""

    global lineCount, procBlock

    file.write("    %s[%d] {pt3dclear()\n" % (node.typeString(), node.buildId))
    for p in node.points:
        file.write(
            "      pt3dadd($1+%.3f, $2+%.3f, $3+%.3f, %.3f)\n"
            % (p[0], p[1], p[2], node.diameter)
        )
    file.write("    }\n")
    lineCount += 1

    # limit proc block length
    if lineCount > 64:
        file.write("  }\n")
        file.write("  proc shape3d_%d() {\n" % (procBlock + 1))
        lineCount = 0
        procBlock += 1

    for c in node.children:
        printShapeBlockNode(c, file)


def printShapeBlock(tree, file):
    """Generate shape-related hoc code in `file'."""

    # reset proc block length counter
    global lineCount, procBlock
    lineCount = 0
    procBlock = 0

    file.write("  proc shape3d_0() {\n")
    printShapeBlockNode(tree, file)
    file.write("  }\n")

    file.write("\n  proc basic_shape() {\n")
    for i in range(procBlock + 1):
        file.write("    shape3d_%d($1, $2, $3)\n" % i)
    file.write("  }\n\n")


def printFooterBlock(modelName, file):
    """Prints footer block to file pointed to by `file."""

    file.write("endtemplate %s\n" % (modelName))


def printInitFile(cellName, file):
    """Print initialization file for the newly created cell."""

    printPreamble(file)
    file.write(
        '{load_file("%s.hoc")}\n'
        '{load_file("global.hoc")}\n'
        "\n"
        "objref cell\n"
        "cell = new %s()\n"
        "\n"
        "//user interface\n"
        '{load_file("gui.hoc")}\n'
        "FIELD_LEFT   = -100\n"
        "FIELD_BOTTOM = -100\n"
        "FIELD_WIDTH  = 300\n"
        "FIELD_HEIGHT = 300\n"
        "// showCell(FIELD_LEFT, FIELD_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT)\n"
        'guiGraph.addvar("cell.soma.v(0.5)", 2, 1)\n'
        'guiGraph.addvar("cell.ais.v(0.5)", 3, 1)\n'
        'guiGraph.addvar("cell.axon.v(0.99)", 4, 1)\n'
        "\n"
        "// instrumentation\n"
        '{load_file("instr.hoc")}\n'
        "\n"
        "// procedures for publication\n"
        "proc pubfig() { local FIELD_LEFT, FIELD_BOTTOM, "
        "FIELD_WIDTH, FIELD_HEIGHT\n"
        "    FIELD_LEFT   = -250\n"
        "    FIELD_BOTTOM = -180\n"
        "    FIELD_WIDTH  = 400\n"
        "    FIELD_HEIGHT = 400\n"
        "    showCell(FIELD_LEFT, FIELD_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT)\n"
        '    morph.printfile("skeleton-off-top.ps")\n'
        "    morph.rotate(0,0,0, PI/2,0,0)\n"
        '    morph.printfile("skeleton-off-side.ps")\n'
        "}\n\n" % (cellName, cellName)
    )


############################################################################
# Main procedure


def main(argv):
    if len(argv) not in [3, 4]:
        print(
            "Usage: generate_hoc [morphology file] [cell name] [OPTIONAL: cell number]\n"
        )
        return

    morphologyFile = argv[1]
    cellName = argv[2]
    cell_no = int(argv[3]) if len(argv) == 4 else 0

    try:
        file = open(morphologyFile)
        tree, cell_type = buildTree(file)
        file.close()
        bridgeDisjointNodes(tree)
        soma_diam = cell_type_to_diameter(cell_type, cell_no)
        fixSingularPointSoma(tree, soma_diam)
        # setNodeDiameter(node, distToRoot, dendDiam, somDiam, hillDiam, aisDiam, axDiam)
        # set for human foveal RGCs
        # soma based on Dacey 1993, Axon based on Watanabe 1989
        ais_diam = 0.6
        # ais_diam = get_ais_diameter(cell_no)  # for testing varying ais diam
        setNodeDiameter(tree, 0, 0.3, soma_diam, 1.25, ais_diam, 0.91)

        output = open(f"{cellName}.hoc", "w")
        printHeaderBlock(cellName, tree, output)
        printBiophysBlock(tree, output)
        printConnectBlock(tree, output)
        printTopolBlock(tree, output)
        printShapeBlock(tree, output)
        printFooterBlock(cellName, output)
        output.close()

        output = open(f"init-{cellName}.hoc", "w")
        printInitFile(cellName, output)
        output.close()

    except:
        raise


if __name__ == "__main__":
    main(sys.argv[:])
