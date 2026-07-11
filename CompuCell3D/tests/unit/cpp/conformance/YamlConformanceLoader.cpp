#include "YamlConformanceLoader.h"

#include <yaml-cpp/yaml.h>

#include <set>
#include <stdexcept>

using namespace CompuCell3D;

namespace {
    [[noreturn]] void fail(const std::string &path, const std::string &message) {
        throw std::runtime_error(path + ": " + message);
    }

    YAML::Node requireNode(const YAML::Node &parent, const char *key, const std::string &path) {
        const YAML::Node child = parent[key];
        if (!child) {
            fail(path, std::string("missing required key '") + key + "'");
        }
        return child;
    }

    ConformancePoint parsePoint(const YAML::Node &node, const std::string &path) {
        if (!node.IsSequence() || node.size() != 3) {
            fail(path, "point must be a 3-element sequence");
        }

        return ConformancePoint{
                node[0].as<int>(),
                node[1].as<int>(),
                node[2].as<int>()
        };
    }

    double parseOptionalDouble(const YAML::Node &node, const char *key, double defaultValue) {
        const YAML::Node child = node[key];
        return child ? child.as<double>() : defaultValue;
    }
}

ConformanceCase CompuCell3D::loadConformanceCase(const std::string &path) {
    const YAML::Node root = YAML::LoadFile(path);

    ConformanceCase testCase;
    testCase.path = path;
    testCase.version = requireNode(root, "version", path).as<int>();
    testCase.name = requireNode(root, "name", path).as<std::string>();
    testCase.domain = requireNode(root, "domain", path).as<std::string>();

    if (testCase.version != 1) {
        fail(path, "only version 1 conformance cases are supported");
    }
    if (testCase.domain != "potts_local") {
        fail(path, "only 'potts_local' domain is supported");
    }

    const YAML::Node latticeNode = requireNode(root, "lattice", path);
    testCase.lattice.dim = parsePoint(requireNode(latticeNode, "dim", path), path + ".lattice.dim");

    const YAML::Node boundaryNode = requireNode(latticeNode, "boundary", path);
    testCase.lattice.boundaryX = requireNode(boundaryNode, "x", path).as<std::string>();
    testCase.lattice.boundaryY = requireNode(boundaryNode, "y", path).as<std::string>();
    testCase.lattice.boundaryZ = requireNode(boundaryNode, "z", path).as<std::string>();

    if (testCase.lattice.boundaryX != "fixed" ||
        testCase.lattice.boundaryY != "fixed" ||
        testCase.lattice.boundaryZ != "fixed") {
        fail(path, "only fixed boundary conditions are supported in the first-pass runner");
    }

    const YAML::Node neighborhoodNode = requireNode(latticeNode, "neighborhood", path);
    testCase.lattice.neighborhoodKind = requireNode(neighborhoodNode, "kind", path).as<std::string>();
    testCase.lattice.neighborhoodOrder = requireNode(neighborhoodNode, "order", path).as<unsigned int>();

    if (testCase.lattice.neighborhoodKind != "moore") {
        fail(path, "only moore neighborhoods are supported in the first-pass runner");
    }

    const YAML::Node mediumNode = root["medium"];
    testCase.mediumType = mediumNode && mediumNode["type"] ? mediumNode["type"].as<std::string>() : "Medium";

    const YAML::Node cellsNode = requireNode(root, "cells", path);
    if (!cellsNode.IsSequence() || cellsNode.size() == 0) {
        fail(path, "cells must be a non-empty sequence");
    }

    std::set<int> cellIds;
    std::set<std::string> cellTypes;
    for (std::size_t i = 0; i < cellsNode.size(); ++i) {
        const YAML::Node cellNode = cellsNode[i];
        ConformanceCell cell;
        cell.id = requireNode(cellNode, "id", path).as<int>();
        cell.type = requireNode(cellNode, "type", path).as<std::string>();
        cell.targetVolume = requireNode(cellNode, "target_volume", path).as<double>();
        cell.lambdaVolume = requireNode(cellNode, "lambda_volume", path).as<double>();
        if (cell.type.empty()) {
            fail(path, "cell type must be non-empty");
        }

        if (!cellIds.insert(cell.id).second) {
            fail(path, "duplicate cell id in cells section");
        }
        cellTypes.insert(cell.type);

        const YAML::Node pixelsNode = requireNode(cellNode, "pixels", path);
        if (!pixelsNode.IsSequence() || pixelsNode.size() == 0) {
            fail(path, "cell pixels must be a non-empty sequence");
        }
        for (std::size_t pixelIndex = 0; pixelIndex < pixelsNode.size(); ++pixelIndex) {
            cell.pixels.push_back(parsePoint(pixelsNode[pixelIndex],
                                             path + ".cells[" + std::to_string(i) + "].pixels[" +
                                             std::to_string(pixelIndex) + "]"));
        }

        testCase.cells.push_back(cell);
    }

    const YAML::Node parametersNode = requireNode(root, "parameters", path);
    const YAML::Node volumeNode = requireNode(parametersNode, "volume", path);
    const std::string volumeKind = requireNode(volumeNode, "kind", path).as<std::string>();
    if (volumeKind != "quadratic") {
        fail(path, "only quadratic volume energy is supported in the first-pass runner");
    }

    const YAML::Node queriesNode = requireNode(root, "queries", path);
    if (!queriesNode.IsSequence() || queriesNode.size() == 0) {
        fail(path, "queries must be a non-empty sequence");
    }

    for (std::size_t i = 0; i < queriesNode.size(); ++i) {
        const YAML::Node queryNode = queriesNode[i];
        const std::string kind = requireNode(queryNode, "kind", path).as<std::string>();
        if (kind != "volume_energy_delta") {
            fail(path, "only volume_energy_delta queries are supported in the first-pass runner");
        }

        ConformanceVolumeQuery query;
        query.id = requireNode(queryNode, "id", path).as<std::string>();
        query.source = parsePoint(requireNode(queryNode, "source", path), path + ".queries[" + std::to_string(i) + "].source");
        query.target = parsePoint(requireNode(queryNode, "target", path), path + ".queries[" + std::to_string(i) + "].target");

        const YAML::Node expectedNode = requireNode(queryNode, "expected", path);
        query.expected.value = requireNode(expectedNode, "value", path).as<double>();
        query.expected.tolerance = parseOptionalDouble(expectedNode, "tolerance", 1.0e-12);

        testCase.volumeQueries.push_back(query);
    }

    if (cellTypes.find(testCase.mediumType) != cellTypes.end()) {
        fail(path, "medium type must not duplicate a cell type");
    }

    return testCase;
}
