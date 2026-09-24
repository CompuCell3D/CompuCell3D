#include <cc3d/kernel.h>

#include <iostream>

namespace {
    // XML handles resolve by ID on each access, so they remain valid across steering updates.
    void start(CC3DKernelContext *ctx, void *) {
        auto temp = ctx->xml.getElement("temp");
        auto volume = ctx->xml.getElement("vol_cond");

        std::cout << "XML steering start: temp="
                  << temp.getValueAsDouble("cdata")
                  << " target="
                  << volume.getValueAsDouble("TargetVolume")
                  << std::endl;
    }

    void step(CC3DKernelContext *ctx, void *) {
        auto temp = ctx->xml.getElement("temp");



        if (ctx->mcs == 10) {
            auto volumeElem = ctx->xml.getElement("vol_cond");
            const double oldTarget = volumeElem.getValueAsDouble("TargetVolume");
            const bool changed = volumeElem.setValue(
                "TargetVolume",
                20.0 * oldTarget
            );

            std::cout << "set volume result=" << changed
                      << " old=" << oldTarget
                      << " value=" << volumeElem.getValueAsDouble("TargetVolume")
                      << std::endl;
        }

        if (ctx->mcs == 20) {
            const bool changed = temp.setValue("cdata", 100.0);

            std::cout << "set temp result=" << changed
                      << " value=" << temp.getValueAsDouble("cdata")
                      << std::endl;
        }

        

        if (ctx->mcs % 10 == 0) {
            std::cout << "XML steering mcs=" << ctx->mcs
                      << " temp=" << temp.getValueAsDouble("cdata")
                      << std::endl;

            // std::cout << "XML steering mcs=" << ctx->mcs
            //           << " targetVolume=" << volumeElem.getValueAsDouble("cdata")
            //           << std::endl;


        }
    }

    void finish(CC3DKernelContext *, void *) {}

    const CC3DSteppableV1 api = {
        CC3D_KERNEL_ABI_VERSION,
        nullptr,
        &start,
        &step,
        &finish,
        nullptr
    };
}

extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
