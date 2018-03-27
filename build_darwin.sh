#!/bin/sh

# For Building on Darwin Platform(macOS)

exename=loop_blinn_ex

cargo build --features ferrite/VK_MVK_macos_surface,ferrite/VK_EXT_debug_report,StemDarkening &&
install_name_tool -change @rpath/vulkan.framework/Versions/A/vulkan @executable_path/vulkan.framework/Versions/A/vulkan target/debug/$exename &&
cp -r $VK_SDK_PATH/macOS/Frameworks/vulkan.framework target/debug/ &&
eval target/debug/$exename
