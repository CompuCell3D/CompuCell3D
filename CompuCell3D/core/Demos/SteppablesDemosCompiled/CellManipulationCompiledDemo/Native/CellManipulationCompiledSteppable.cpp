#include <cc3d/kernel.h>

#include <iostream>

namespace {
    struct State {
        long sourceId = -1;
        long destinationId = -1;
    };

    void *create() { return new State; }

    void start(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        const auto condensing = ctx->cellTypes["Condensing"];

        for (auto cell : ctx->cells.byType(static_cast<unsigned char>(condensing))) {
            state->sourceId = cell.id;
            break;
        }

        std::cerr << "CellManipulation start source=" << state->sourceId << '\n';
    }

    void step(CC3DKernelContext *ctx, void *statePtr) {
        auto *state = static_cast<State *>(statePtr);
        auto manipulation = ctx->cellManipulation.cellManipulation();

        if (ctx->mcs == 10 && state->sourceId >= 0) {
            auto source = ctx->cells.fetch_cell_by_id(state->sourceId);
            if (source && manipulation.moveCell(source.value(), 20, 20, 0)) {
                std::cerr << "CellManipulation moved cell=" << state->sourceId << '\n';
            }
        }

        if (ctx->mcs == 20) {
            const auto nonCondensing = ctx->cellTypes["NonCondensing"];
            auto destination = ctx->newCell(static_cast<unsigned char>(nonCondensing));
            ctx->cellField.assign(50, 55, 50, 55, 0, 1, destination);
            state->destinationId = destination.id;
            std::cerr << "CellManipulation created destination=" << state->destinationId << '\n';
        }

        if (ctx->mcs == 30 && state->sourceId >= 0 && state->destinationId >= 0) {
            auto source = ctx->cells.fetch_cell_by_id(state->sourceId);
            auto destination = ctx->cells.fetch_cell_by_id(state->destinationId);
            if (source && destination && manipulation.mergeCells(source.value(), destination.value())) {
                std::cerr << "CellManipulation merged source=" << state->sourceId
                          << " into destination=" << state->destinationId << '\n';
            }
        }

        if (ctx->mcs == 40 && state->destinationId >= 0) {
            auto destination = ctx->cells.fetch_cell_by_id(state->destinationId);
            if (destination && manipulation.deleteCell(destination.value())) {
                std::cerr << "CellManipulation deleted cell=" << state->destinationId << '\n';
            }
        }
    }

    void finish(CC3DKernelContext *, void *) {}
    void destroy(void *state) { delete static_cast<State *>(state); }

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION, &create, &start, &step, &finish, &destroy
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
