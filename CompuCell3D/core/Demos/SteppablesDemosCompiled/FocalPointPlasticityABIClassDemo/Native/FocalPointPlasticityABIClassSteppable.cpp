// Use the repository ABI header directly. The active Python environment may
// contain an older installed header without the FPP inventory operations.
#include <cc3d/kernel.h>
#include <iostream>

namespace {

class FocalPointPlasticitySteppable {
public:
    void start(CC3DKernelContext *ctx) {
        const auto cell1Type = ctx->cellTypes["Cell1"];
        const auto cell2Type = ctx->cellTypes["Cell2"];

        auto initiator = ctx->newCell(static_cast<unsigned char>(cell1Type));
        initiator.targetVolume = 120.0f;
        initiator.lambdaVolume = 2.0f;
        ctx->cellField.assign(8, 20, 10, 20, 0, 1, initiator);
        initiatorId_ = initiator.id;

        auto initiated = ctx->newCell(static_cast<unsigned char>(cell2Type));
        initiated.targetVolume = 120.0f;
        initiated.lambdaVolume = 2.0f;
        ctx->cellField.assign(40, 52, 10, 20, 0, 1, initiated);
        initiatedId_ = initiated.id;
    }

    void step(CC3DKernelContext *ctx) {
        if (!ctx) return;

        auto initiatorResult = ctx->cells.fetch_cell_by_id(initiatorId_);
        auto initiatedResult = ctx->cells.fetch_cell_by_id(initiatedId_);
        if (!initiatorResult || !initiatedResult) return;

        auto initiator = initiatorResult.value();
        auto initiated = initiatedResult.value();
        auto fpp = ctx->focalPointPlasticityQueries.focalPointPlasticity(
            &ctx->focalPointPlasticity);

        if (ctx->mcs == 0u) {
            fpp.create(initiator, initiated, 10.0, 15.0, 40.0);
            fpp.createInternal(initiator, initiated, 7.0, 12.0, 35.0);
            fpp.createAnchor(initiator, 5.0, 8.0, 30.0, 20.0f, 15.0f, 0.0f);
        }

        if (ctx->mcs == 1u) {
            std::cerr << "FPP counts: links=" << fpp.linkCount()
                      << " internal=" << fpp.internalLinkCount()
                      << " anchors=" << fpp.anchorCount() << std::endl;

            std::cerr << "Per-cell counts for cell "
                      << static_cast<long>(initiator.id)
                      << ": links=" << fpp.linkCount(initiator)
                      << " internal=" << fpp.internalLinkCount(initiator)
                      << " anchors=" << fpp.anchorCount(initiator) << std::endl;

            for (auto link : fpp.links()) {
                std::cerr << "Regular link: "
                          << static_cast<long>(link.initiator().id) << " -> "
                          << static_cast<long>(link.initiated().id) << std::endl;
            }

            for (auto link : fpp.internalLinks(initiator)) {
                std::cerr << "Internal link: "
                          << static_cast<long>(link.initiator().id) << " -> "
                          << static_cast<long>(link.initiated().id) << std::endl;
            }

            for (auto anchor : fpp.anchors(initiator)) {
                std::cerr << "Anchor " << anchor.anchorId()
                          << " on cell " << static_cast<long>(anchor.initiator().id)
                          << std::endl;
            }

            auto link = fpp.findLink(initiator, initiated);
            if (link) {
                std::cerr << "Pair lookup found link from "
                          << static_cast<long>(link.initiator().id) << " to "
                          << static_cast<long>(link.initiated().id) << std::endl;
            }

            for (auto cell : fpp.linkedCells(initiator)) {
                std::cerr << "Linked cell: " << static_cast<long>(cell.id) << std::endl;
            }

            for (auto cell : fpp.internalLinkedCells(initiator)) {
                std::cerr << "Internally linked cell: "
                          << static_cast<long>(cell.id) << std::endl;
            }
        }
    }

private:
    long initiatorId_ = -1;
    long initiatedId_ = -1;
};

void *create() { return new FocalPointPlasticitySteppable(); }

void start(CC3DKernelContext *ctx, void *state) {
    static_cast<FocalPointPlasticitySteppable *>(state)->start(ctx);
}

void step(CC3DKernelContext *ctx, void *state) {
    static_cast<FocalPointPlasticitySteppable *>(state)->step(ctx);
}

void destroy(void *state) {
    delete static_cast<FocalPointPlasticitySteppable *>(state);
}

}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    static const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, nullptr, &destroy
    };
    return &api;
}
