#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <iostream>
#include <string>
#include <exception>
#include <vector>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <set>
#include <limits>
#include <algorithm>
#include <fstream>

#define VDEBUG

class TriangleApplication {

    struct QueueFamilyIndices {
        std::optional<uint32_t> _grapics_family;
        std::optional<uint32_t> _present_family;

        bool is_complete()          const   { return _grapics_family.has_value() &&
                                                     _present_family.has_value();   }

        uint32_t graphics_family()  const   { return _grapics_family.value();       }
        void set_graphics_family(uint32_t f){ _grapics_family = f;                  }
        uint32_t present_family()   const   { return _present_family.value();       }
        void set_present_family(uint32_t f) { _present_family = f;                  }
    };

    struct SwapchainSupportDetails {
        VkSurfaceCapabilitiesKHR        capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   present_mods;
    };

    VkInstance  _vk_instance;
    GLFWwindow* _window;
    VkPhysicalDevice            _physical_device;
    VkDevice                    _logical_device;
    VkQueue                     _graphics_queue;
    VkQueue                     _present_queue;
    VkDebugUtilsMessengerEXT    _debug_messenger;
    VkSurfaceKHR                _surface;
    VkSwapchainKHR              _swapchain;
    std::vector<VkImage>        _swapchain_images;
    std::vector<VkImageView>    _swapchain_image_views;
    std::vector<VkFramebuffer>  _framebuffers;
    VkExtent2D                  _swapchain_extent;
    VkSurfaceFormatKHR          _swapchain_image_format;
    VkPipelineLayout            _pipe_layout;
    VkPipeline                  _graphics_pipeline;
    VkRenderPass                _render_pass;
    VkCommandPool               _command_pool;
    std::vector<VkCommandBuffer>_command_buffers;
    std::vector<VkSemaphore>    _image_available_semaphores;
    std::vector<VkSemaphore>    _render_finished_semaphores;
    std::vector<VkFence>        _in_flight_flences;
    size_t                      _current_frame;
    bool                        _window_resizable;

    const int _window_width     = 800;
    const int _window_height    = 600;
    const int _max_frames       = 2;

    #ifdef VDEBUG
    bool _validation_layers_enabled = true;
    #elif 
    bool _validation_layers_enabled = false;
    #endif

    std::vector<const char*> _validation_layers;
    std::vector<const char*> _required_extensions;
    std::vector<const char*> _device_extensions;

    void init_vulkan(const std::string& app_name);
    void init_window(const std::string& app_name);
    void init_instance(const std::string& app_name);
    void setup_debug();
    void create_surface();
    void pick_physical_device();
    void create_logical_device();
    void create_swapchain();
    void recreate_swapchain();
    void create_image_views();
    void create_pipelines();
    void create_render_pass();
    void create_framebuffers();
    void create_command_pool();
    void create_command_buffer();
    void create_syncs();
    void main_loop();

    bool device_is_suitable(VkPhysicalDevice device) const;
    QueueFamilyIndices find_queue_family(VkPhysicalDevice device) const;
    bool check_validation_layers_supported() const;
    bool device_extension_supported(VkPhysicalDevice device) const;
    SwapchainSupportDetails query_support_swapchain(VkPhysicalDevice device, VkSurfaceKHR surface) const;
    VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) const;
    VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& mods) const ;
    VkExtent2D choose_swapchain_extent(const VkSurfaceCapabilitiesKHR& capabilities) const;
    std::vector<char> read_file(const std::string& file_path) const;
    void record_command_buffer(VkCommandBuffer buffer, uint32_t imageIndex);
    void draw_frame();

    void destroy_vulkan();
    void destroy_window();
    void destroy_debug();
    void destroy_surface();
    void destroy_logical_device();
    void destroy_swapchain();
    void cleanup_swapchain();
    void destroy_image_views();
    void destroy_layout();
    void destroy_render_pass();
    void destroy_framebuffers();
    void destroy_command_pool();
    void destroy_syncs();

    VkApplicationInfo               application_info(const std::string& app_name) const;
    VkInstanceCreateInfo            instance_create_info(const VkApplicationInfo* app_info) const;
    VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info() const;
    VkDeviceQueueCreateInfo         device_queue_create_info(uint32_t indices, uint32_t count, float* priorities) const;
    VkDeviceCreateInfo              device_create_info(const std::vector<VkDeviceQueueCreateInfo>& queue_create_info, const VkPhysicalDeviceFeatures* featues) const;
    VkSwapchainCreateInfoKHR        swapchain_create_info(VkPhysicalDevice device, 
                                                          const SwapchainSupportDetails& details,
                                                          VkSurfaceKHR surface) const;
    VkImageViewCreateInfo           image_view_create_info(const VkImage& image) const;
    VkShaderModuleCreateInfo        shader_module_create_info(std::vector<char>& module_data) const;
    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(const VkShaderModule& module, VkShaderStageFlagBits stage) const;
    VkRenderPassCreateInfo          render_pass_create_info(const VkAttachmentDescription& description, const VkSubpassDescription& subpass, const std::vector<VkSubpassDependency>& subpass_deps) const;

    VkResult    create_debug_utils_messenger_ext(VkInstance instance, 
                                                 const VkDebugUtilsMessengerCreateInfoEXT* messenger_create_info,
                                                 const VkAllocationCallbacks* callback,
                                                 VkDebugUtilsMessengerEXT* messenger);
    VkResult    destroy_debug_utils_messenger_ext(VkInstance instance,
                                                  VkDebugUtilsMessengerEXT messenger,
                                                  const VkAllocationCallbacks* callback);
    

    void populate_required_extensions();
    void populate_validation_layers();
    void populate_device_extensions();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT m_severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT m_type,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* p_data,
                                                         void* user_data)
    {
        if (m_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT){
            std::cout   << "VULKAN_ERROR |\n"
                        << "\t\tm_severity{" << m_severity << "}\n"
                        << "\t\tm_type{" << m_type << "}\n"
                        << "\t\tmessage{" << p_data->pMessage << "}"
                        << std::endl << std::endl << std::endl;
        }
        return VK_FALSE;
    }

    static void window_size_changed_callback(GLFWwindow* window, int, int) {
        auto app = reinterpret_cast<TriangleApplication*>(glfwGetWindowUserPointer(window));
        app->_window_resizable = true;
    }

public:
    TriangleApplication(const std::string& app_name);
    ~TriangleApplication();
    void run();
};

constexpr const char* default_appname = "Application";

int main(int argc, char* args[]){
    std::string app_name(argc > 1 ? args[1] : default_appname);
    try{
        TriangleApplication(app_name).run();
    } catch(const std::exception& excp){
        std::cerr << excp.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

TriangleApplication::TriangleApplication(const std::string& app_name) :
    _vk_instance{VK_NULL_HANDLE}
    ,_physical_device{VK_NULL_HANDLE}
    ,_window{nullptr}
    ,_current_frame{0}
{ 
    init_window(app_name);
    init_vulkan(app_name);
}

TriangleApplication::~TriangleApplication() {
    destroy_window();
    destroy_vulkan();
}

void TriangleApplication::init_window(const std::string& app_name){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    _window = glfwCreateWindow(_window_width, _window_height, app_name.data(), nullptr, nullptr);
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, window_size_changed_callback);
    if (!_window){
        throw std::runtime_error("Window don't created");
    }
}

void TriangleApplication::init_vulkan(const std::string& app_name){
    populate_required_extensions();
    populate_validation_layers();
    populate_device_extensions();
    init_instance(app_name);
    setup_debug();
    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_pipelines();
    create_framebuffers();
    create_command_pool();
    create_command_buffer();
    create_syncs();
}

void TriangleApplication::init_instance(const std::string& app_name){
    if (_validation_layers_enabled && !check_validation_layers_supported()){
        throw std::runtime_error("Requested layers don't supported");
    }
    auto app_info = application_info(app_name);
    auto instance_cinfo = instance_create_info(&app_info);
    if (vkCreateInstance(&instance_cinfo, nullptr, &_vk_instance) != VK_SUCCESS) {
        throw std::runtime_error("VKInstance don't created");
    }
}

void TriangleApplication::setup_debug(){
    auto messenger_cinfo = debug_utils_messenger_create_info();
    if (create_debug_utils_messenger_ext(_vk_instance, &messenger_cinfo, nullptr, &_debug_messenger) != VK_SUCCESS){
        throw std::runtime_error("setup_debug DEBUG_UTILS extension not present");
    }
}

void TriangleApplication::create_surface() {
    if (glfwCreateWindowSurface(_vk_instance, _window, nullptr, &_surface) != VK_SUCCESS){
        throw std::runtime_error("Surface don't created");
    }
}

void TriangleApplication::pick_physical_device(){
    uint32_t    device_count;
    vkEnumeratePhysicalDevices(_vk_instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice>   devices(device_count);
    vkEnumeratePhysicalDevices(_vk_instance, &device_count, devices.data());

    for (const auto& device: devices){
        if (device_is_suitable(device)) _physical_device = device;
    }
    if (_physical_device == VK_NULL_HANDLE){
        throw std::runtime_error("Suitable device not found");
    }

    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties (_physical_device, &prop);
    std::cout << "Selected physical device:" << prop.deviceName << " with type " << prop.deviceType << std::endl;
}

void TriangleApplication::create_logical_device() {
    auto family_indices = find_queue_family(_physical_device);

    uint32_t count = 1;
    float priorities = 1.0f;
    VkPhysicalDeviceFeatures features = {};
    
    std::set<uint32_t> families = {family_indices.graphics_family(), family_indices.present_family()};
    std::vector<VkDeviceQueueCreateInfo> infos;
    for (auto family: families){
        auto queue_cinfo = device_queue_create_info(family, count, &priorities);
        infos.push_back(queue_cinfo);
    }
    auto device_cinfo = device_create_info(infos, &features);
    vkCreateDevice(_physical_device, &device_cinfo, nullptr, &_logical_device);
    vkGetDeviceQueue(_logical_device, family_indices.graphics_family(), 0, &_graphics_queue);
    vkGetDeviceQueue(_logical_device, family_indices.present_family(), 0, &_present_queue);
}

void TriangleApplication::create_swapchain() {
    auto details = query_support_swapchain(_physical_device, _surface);
    auto swapchain_cinfo = swapchain_create_info(_physical_device, details, _surface);
    if (vkCreateSwapchainKHR(_logical_device, &swapchain_cinfo, nullptr, &_swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Swapchaint don't created");
    }
    
    uint32_t    image_count; 
    vkGetSwapchainImagesKHR(_logical_device, _swapchain, &image_count, nullptr);
    _swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(_logical_device, _swapchain, &image_count, _swapchain_images.data());

    _swapchain_extent = choose_swapchain_extent(details.capabilities);
    _swapchain_image_format = choose_surface_format(details.formats);
}

void TriangleApplication::recreate_swapchain() {
    int width, height;
    glfwGetWindowSize(_window, &width, &height);
    if (width == 0 || height == 0){
        glfwGetWindowSize(_window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(_logical_device);

    cleanup_swapchain();
    
    create_swapchain();
    create_image_views();
    create_framebuffers();
}

void TriangleApplication::create_image_views() {
    _swapchain_image_views.resize(_swapchain_images.size());
    for (size_t ind; ind < _swapchain_images.size(); ind++){
        auto view_cinfo = image_view_create_info(_swapchain_images[ind]);
        vkCreateImageView(_logical_device, &view_cinfo, nullptr, &_swapchain_image_views[ind]);
    }
}

void TriangleApplication::create_pipelines() {
    auto vert_data = read_file("shaders/vert.spv");
    auto frag_data = read_file("shaders/frag.spv");
    auto vmodule_cinfo = shader_module_create_info(vert_data);
    auto fmodule_cinfo = shader_module_create_info(frag_data);

    VkShaderModule  vmodule = {};
    VkShaderModule  fmodule = {};
    if (vkCreateShaderModule(_logical_device, &vmodule_cinfo, nullptr, &vmodule) != VK_SUCCESS){
        throw std::runtime_error("Vertext shader module don't created");
    }
    if (vkCreateShaderModule(_logical_device, &fmodule_cinfo, nullptr, &fmodule) != VK_SUCCESS){
        throw std::runtime_error("Fragment shader module don't created");
    }

    auto pipe_vm_cinfo = pipeline_shader_stage_create_info(vmodule, VK_SHADER_STAGE_VERTEX_BIT);
    auto pipe_fm_cinfo = pipeline_shader_stage_create_info(fmodule, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineShaderStageCreateInfo shader_stages[] = {pipe_vm_cinfo, pipe_fm_cinfo};

    std::vector<VkDynamicState> states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state_cinfo = {};
    dynamic_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state_cinfo.dynamicStateCount = static_cast<uint32_t>(states.size());
    dynamic_state_cinfo.pDynamicStates = states.data();

    VkPipelineVertexInputStateCreateInfo vi_state_cinfo = {};
    vi_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi_state_cinfo.vertexBindingDescriptionCount = 0;
    vi_state_cinfo.pVertexBindingDescriptions = nullptr;
    vi_state_cinfo.vertexAttributeDescriptionCount = 0;
    vi_state_cinfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo ia_state_cinfo = {};
    ia_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia_state_cinfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    ia_state_cinfo.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport;
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float) _swapchain_extent.width;
    viewport.height = (float) _swapchain_extent.height;
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    VkRect2D scissor;
    scissor.offset = {0, 0};
    scissor.extent = _swapchain_extent;

    VkPipelineViewportStateCreateInfo vp_state_cinfo = {};
    vp_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp_state_cinfo.scissorCount = 1;
    vp_state_cinfo.pScissors = &scissor;
    vp_state_cinfo.viewportCount = 1;
    vp_state_cinfo.pViewports = &viewport;

    VkPipelineRasterizationStateCreateInfo r_state_cinfo = {};
    r_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    r_state_cinfo.depthClampEnable = VK_FALSE;
    r_state_cinfo.rasterizerDiscardEnable = VK_FALSE;
    r_state_cinfo.cullMode = VK_CULL_MODE_BACK_BIT;
    r_state_cinfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
    r_state_cinfo.depthBiasEnable = VK_FALSE;
    r_state_cinfo.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms_state_cinfo = {};
    ms_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms_state_cinfo.sampleShadingEnable = VK_FALSE;
    ms_state_cinfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba_state = {};
    cba_state.colorWriteMask    = VK_COLOR_COMPONENT_R_BIT |
                                  VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT;
    cba_state.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb_state_cinfo = {};
    cb_state_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb_state_cinfo.logicOpEnable = VK_FALSE;
    cb_state_cinfo.logicOp = VK_LOGIC_OP_COPY;
    cb_state_cinfo.attachmentCount = 1;
    cb_state_cinfo.pAttachments = &cba_state;

    VkPipelineLayoutCreateInfo layout_cinfo = {};
    layout_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_cinfo.setLayoutCount = 0;
    layout_cinfo.pSetLayouts = nullptr;
    layout_cinfo.pushConstantRangeCount = 0;
    layout_cinfo.pPushConstantRanges = nullptr;
    if (vkCreatePipelineLayout(_logical_device, &layout_cinfo, nullptr, &_pipe_layout) != VK_SUCCESS){
        throw std::runtime_error("Pipeline layout don't created");
    }

    VkGraphicsPipelineCreateInfo gpipeline_cinfo = {};
    gpipeline_cinfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpipeline_cinfo.stageCount = 2;
    gpipeline_cinfo.pStages = shader_stages;
    gpipeline_cinfo.pInputAssemblyState = &ia_state_cinfo;
    gpipeline_cinfo.pVertexInputState = &vi_state_cinfo;
    gpipeline_cinfo.pRasterizationState = &r_state_cinfo;
    gpipeline_cinfo.pMultisampleState = &ms_state_cinfo;
    gpipeline_cinfo.pDynamicState = nullptr;//&dynamic_state_cinfo;
    gpipeline_cinfo.pViewportState = &vp_state_cinfo;
    gpipeline_cinfo.pColorBlendState = &cb_state_cinfo;
    gpipeline_cinfo.renderPass = _render_pass;
    gpipeline_cinfo.subpass = 0;
    gpipeline_cinfo.layout = _pipe_layout;
    if (vkCreateGraphicsPipelines(_logical_device, VK_NULL_HANDLE, 1, &gpipeline_cinfo, nullptr, &_graphics_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Graphics pipeline don't created");
    }

    vkDestroyShaderModule(_logical_device, vmodule, nullptr);
    vkDestroyShaderModule(_logical_device, fmodule, nullptr);
}

void TriangleApplication::create_framebuffers() {
    _framebuffers.resize(_swapchain_image_views.size());

    for (size_t i = 0; i < _swapchain_image_views.size(); i++){
        VkFramebufferCreateInfo framebuffer_cinfo = {};
        framebuffer_cinfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_cinfo.renderPass = _render_pass;
        framebuffer_cinfo.height = _swapchain_extent.height;
        framebuffer_cinfo.width = _swapchain_extent.width;
        framebuffer_cinfo.attachmentCount = 1;
        framebuffer_cinfo.pAttachments = &_swapchain_image_views[i];
        framebuffer_cinfo.layers = 1;
        if (vkCreateFramebuffer(_logical_device, &framebuffer_cinfo, nullptr, &_framebuffers[i]) != VK_SUCCESS){
            throw std::runtime_error("Framebuffer don't created");
        }
    }
}

void TriangleApplication::create_command_pool() {
    auto families = find_queue_family(_physical_device);

    VkCommandPoolCreateInfo pool_cinfo = {};
    pool_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_cinfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_cinfo.queueFamilyIndex = families.graphics_family();

    if (vkCreateCommandPool(_logical_device, &pool_cinfo, nullptr, &_command_pool) != VK_SUCCESS) {
        throw std::runtime_error("Command pool don't created");
    }
}

void TriangleApplication::create_command_buffer() {
    _command_buffers.resize(_max_frames);
    for (size_t i = 0; i < _max_frames; i++){
        VkCommandBufferAllocateInfo buffer_ainfo = {};
        buffer_ainfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        buffer_ainfo.commandBufferCount = 1;
        buffer_ainfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        buffer_ainfo.commandPool = _command_pool;
        if (vkAllocateCommandBuffers(_logical_device, &buffer_ainfo, &_command_buffers[i]) != VK_SUCCESS){
            throw std::runtime_error("Command buffer don't allocated");
        }
    }
}

void TriangleApplication::create_syncs() {
    _in_flight_flences.resize(_max_frames);
    _image_available_semaphores.resize(_max_frames);
    _render_finished_semaphores.resize(_max_frames);
    
    for (size_t i = 0; i < _max_frames; i++){
        VkSemaphoreCreateInfo image_scinfo = {};
        VkSemaphoreCreateInfo render_scinfo = {};
        VkFenceCreateInfo fence_cinfo = {};
        image_scinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        render_scinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        fence_cinfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_cinfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        std::cout << __PRETTY_FUNCTION__ << " i:" << i << std::endl;

        if 
            (
            vkCreateFence(_logical_device, &fence_cinfo, nullptr, &_in_flight_flences[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_logical_device, &image_scinfo, nullptr, &_image_available_semaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_logical_device, &render_scinfo, nullptr, &_render_finished_semaphores[i]) != VK_SUCCESS
            ) 
        {
            throw std::runtime_error("Syncs don't created");
        }
    }
}

void TriangleApplication::create_render_pass() {
    VkAttachmentDescription color_desc = {};
    color_desc.format = _swapchain_image_format.format;
    color_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    color_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_desc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_desc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref = {};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;

    VkSubpassDependency subpass_dep = {};
    subpass_dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpass_dep.dstSubpass = 0;
    subpass_dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dep.srcAccessMask = 0;
    subpass_dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::vector<VkSubpassDependency> subpass_deps = {subpass_dep};

    VkSubpassDescription pass_ref = {};
    pass_ref.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    pass_ref.colorAttachmentCount = 1;
    pass_ref.pColorAttachments = &color_ref;

    auto renderpass_cinfo = render_pass_create_info(color_desc, pass_ref, subpass_deps);
    if (vkCreateRenderPass(_logical_device, &renderpass_cinfo, nullptr, &_render_pass) != VK_SUCCESS){
        throw std::runtime_error("RenderPass don't created");
    }
}

bool TriangleApplication::check_validation_layers_supported() const {
    uint32_t    layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties>  layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

    for (auto requested_layer: _validation_layers){
        bool have = false;
        
        for (auto supported_layer: layers){
            if (strcmp(requested_layer, supported_layer.layerName) == 0){
                have = true;
            }
        }

        if (!have) return false;
    }

    return true;
}

bool TriangleApplication::device_extension_supported(VkPhysicalDevice device) const {
    uint32_t    ext_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
    std::set<std::string> req_exts(_device_extensions.begin(), _device_extensions.end());
    std::cout << __PRETTY_FUNCTION__ << " " << ext_count << " " << req_exts.size() << "\n";
    if (ext_count < static_cast<uint32_t>(req_exts.size())) return false;
    std::vector<VkExtensionProperties> props(ext_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, props.data());
    std::cout << __PRETTY_FUNCTION__ << " after return " << props.size() << "\n";
    for (auto prop: props){
        std::cout << prop.extensionName << " supported extension\n";
        req_exts.erase(prop.extensionName);
    }
    return req_exts.empty();
}

TriangleApplication::SwapchainSupportDetails TriangleApplication::query_support_swapchain(VkPhysicalDevice device, VkSurfaceKHR surface) const {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    
    uint32_t    format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0){
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }

    uint32_t    mod_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mod_count, nullptr);
    if (mod_count != 0) {
        details.present_mods.resize(mod_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mod_count, details.present_mods.data());
    }

    return details;
}

VkSurfaceFormatKHR TriangleApplication::choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) const {
    for (auto surface: formats){
        if (surface.format == VK_FORMAT_B8G8R8A8_SRGB &&
            surface.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return surface;
    }
    return formats[0];
}

VkExtent2D TriangleApplication::choose_swapchain_extent(const VkSurfaceCapabilitiesKHR& capabilities) const {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
        return capabilities.currentExtent;
    }

    int window_width = 0;
    int window_height = 0;
    glfwGetFramebufferSize(_window, &window_width, &window_height);

    VkExtent2D extent{
        static_cast<uint32_t>(window_width), 
        static_cast<uint32_t>(window_height)
    };

    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

std::vector<char> TriangleApplication::read_file(const std::string& file_path) const {
    std::ifstream file(file_path, std::ios::ate | std::ios::binary);
    if (!file.is_open()){
        throw std::runtime_error("File " + file_path + " don't opened");
    }
    size_t  file_size = file.tellg();
    std::vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    std::cout << __PRETTY_FUNCTION__ << " file size: " << file_size << " bufer_size: " << buffer.size() << std::endl;
    return buffer;
}

void TriangleApplication::record_command_buffer(VkCommandBuffer buffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo buffer_binfo = {};
    buffer_binfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(buffer, &buffer_binfo) != VK_SUCCESS){
        throw std::runtime_error("Don't going to begin command buffer");
    }

    VkRenderPassBeginInfo render_binfo = {};
    render_binfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_binfo.renderPass = _render_pass;

    std::cout << __PRETTY_FUNCTION__ << " imageIndex:" << imageIndex << std::endl;

    render_binfo.framebuffer = _framebuffers[imageIndex];
    render_binfo.renderArea.offset = {0, 0};
    render_binfo.renderArea.extent = _swapchain_extent;

    VkClearValue clear_value = {{{0.0, 0.0, 0.0, 1.0}}};
    render_binfo.clearValueCount = 1;
    render_binfo.pClearValues = &clear_value;

    vkCmdBeginRenderPass(buffer, &render_binfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics_pipeline);
    vkCmdDraw(buffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(buffer);
    if (vkEndCommandBuffer(buffer) != VK_SUCCESS){
        throw std::runtime_error("");
    }
}

void TriangleApplication::draw_frame() {
    vkWaitForFences(_logical_device, 1, &_in_flight_flences[_current_frame], VK_TRUE, UINT64_MAX);
    vkResetFences(_logical_device, 1, &_in_flight_flences[_current_frame]);

    uint32_t imageIndex;
    auto result = vkAcquireNextImageKHR(_logical_device, _swapchain, UINT64_MAX, _image_available_semaphores[_current_frame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || _window_resizable) {
        recreate_swapchain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Next image acquire error");
    }

    vkResetCommandBuffer(_command_buffers[_current_frame], /*VkCommandBufferResetFlagBits*/ 0);
    record_command_buffer(_command_buffers[_current_frame], imageIndex);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {_image_available_semaphores[_current_frame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_command_buffers[_current_frame];

    VkSemaphore signalSemaphores[] = {_render_finished_semaphores[_current_frame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(_graphics_queue, 1, &submitInfo, _in_flight_flences[_current_frame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {_swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(_present_queue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Error queue present");
    }

    _current_frame = (_current_frame + 1) % _max_frames;
}

VkPresentModeKHR TriangleApplication::choose_present_mode(const std::vector<VkPresentModeKHR>& mods) const {
    for (const auto& mod: mods){
        if (mod == VK_PRESENT_MODE_MAILBOX_KHR){
            return mod;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

void TriangleApplication::main_loop() {
    while(!glfwWindowShouldClose(_window)){
        glfwPollEvents();
        draw_frame();
    }

    vkDeviceWaitIdle(_logical_device);
}

bool TriangleApplication::device_is_suitable(VkPhysicalDevice device) const {
    bool extension_supported = device_extension_supported(device);
    std::cout << __PRETTY_FUNCTION__ << std::boolalpha << " supported: " << extension_supported << std::endl;
    bool swapchain_supported = false;
    if (extension_supported){
        auto details = query_support_swapchain(device, _surface);
        swapchain_supported = !(details.formats.empty() || details.present_mods.empty());
        std::cout << __PRETTY_FUNCTION__ <<  " " << std::boolalpha << details.formats.empty() << " " << details.present_mods.empty() << std::endl;
    }

    return find_queue_family(device).is_complete() && extension_supported && swapchain_supported;
}

TriangleApplication::QueueFamilyIndices TriangleApplication::find_queue_family(VkPhysicalDevice device) const {
    uint32_t    queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    QueueFamilyIndices indices;

    for (uint32_t i = 0; i < (uint32_t) queue_families.size(); i++){
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT){
            indices.set_graphics_family(i);
        }

        VkBool32 surface_supported;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &surface_supported);
        if (surface_supported)  indices.set_present_family(i);
        if (indices.is_complete()) return indices;
    }

    return indices;
}

void TriangleApplication::destroy_window() {
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void TriangleApplication::destroy_vulkan() {
    destroy_syncs();
    destroy_command_pool();
    destroy_framebuffers();
    destroy_layout();
    destroy_render_pass();
    destroy_image_views();
    destroy_swapchain();
    destroy_surface();
    destroy_debug();
    destroy_logical_device();
    vkDestroyInstance(_vk_instance, nullptr);
}

void TriangleApplication::destroy_syncs() {
    for (size_t i = 0; i < _max_frames; i++){
        vkDestroySemaphore(_logical_device, _render_finished_semaphores[i], nullptr);
        vkDestroySemaphore(_logical_device, _image_available_semaphores[i], nullptr);
        vkDestroyFence(_logical_device, _in_flight_flences[i], nullptr);
    }
}

void TriangleApplication::destroy_command_pool() {
    vkDestroyCommandPool(_logical_device, _command_pool, nullptr);
}

void TriangleApplication::destroy_framebuffers() {
    for(auto framebuffer: _framebuffers) {
        vkDestroyFramebuffer(_logical_device, framebuffer, nullptr);
    }
}

void TriangleApplication::destroy_surface(){
    vkDestroySurfaceKHR(_vk_instance, _surface, nullptr);
}

void TriangleApplication::destroy_debug() {
    if (!_validation_layers_enabled) return;
    if (destroy_debug_utils_messenger_ext(_vk_instance, _debug_messenger, nullptr) != VK_SUCCESS){
        throw std::runtime_error("DEBUG_UTILS extension not present");
    }
}

void TriangleApplication::destroy_logical_device() {
    vkDestroyDevice(_logical_device, nullptr);
}

void TriangleApplication::destroy_swapchain() {
    vkDestroySwapchainKHR(_logical_device, _swapchain, nullptr);
}

void TriangleApplication::cleanup_swapchain() {
    destroy_framebuffers();
    destroy_image_views();
    destroy_swapchain();
}

void TriangleApplication::destroy_image_views() {
    for (auto view: _swapchain_image_views){
        vkDestroyImageView(_logical_device, view, nullptr);
    }
}

void TriangleApplication::destroy_render_pass() {
    vkDestroyRenderPass(_logical_device, _render_pass, nullptr);
}

void TriangleApplication::destroy_layout() {
    vkDestroyPipeline(_logical_device, _graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(_logical_device, _pipe_layout, nullptr);
}

VkApplicationInfo TriangleApplication::application_info(const std::string& app_name) const {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name.data();
    app_info.applicationVersion = VK_MAKE_VERSION(1,0,0);
    app_info.pEngineName = "No engine";
    app_info.engineVersion = VK_MAKE_VERSION(1,0,0);
    app_info.apiVersion = VK_API_VERSION_1_0;
    return app_info;
}

VkInstanceCreateInfo TriangleApplication::instance_create_info(const VkApplicationInfo* app_info) const {
    VkInstanceCreateInfo instance_cinfo = {};
    instance_cinfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_cinfo.pApplicationInfo = app_info;

    instance_cinfo.enabledExtensionCount    = static_cast<uint32_t>(_required_extensions.size());
    instance_cinfo.ppEnabledExtensionNames  = _required_extensions.data();

    if (_validation_layers_enabled){
        instance_cinfo.enabledLayerCount    = static_cast<uint32_t>(_validation_layers.size());
        instance_cinfo.ppEnabledLayerNames  = _validation_layers.data();
    }   

    return instance_cinfo;
}

VkDebugUtilsMessengerCreateInfoEXT TriangleApplication::debug_utils_messenger_create_info()  const {
    VkDebugUtilsMessengerCreateInfoEXT messenger_cinfo = {};
    messenger_cinfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messenger_cinfo.messageSeverity     = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_cinfo.messageType         = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_cinfo.pUserData = nullptr;
    messenger_cinfo.pfnUserCallback = debug_callback;
    return messenger_cinfo;
}

VkDeviceQueueCreateInfo TriangleApplication::device_queue_create_info(uint32_t family_indices, uint32_t count, float* priorities) const {
    VkDeviceQueueCreateInfo queue_cinfo = {};
    queue_cinfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_cinfo.queueFamilyIndex = family_indices; 
    queue_cinfo.queueCount = count;
    queue_cinfo.pQueuePriorities = priorities;
    return queue_cinfo;
}

VkDeviceCreateInfo TriangleApplication::device_create_info(const std::vector<VkDeviceQueueCreateInfo>& infos, const VkPhysicalDeviceFeatures* features) const {
    VkDeviceCreateInfo device_cinfo = {};
    device_cinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_cinfo.queueCreateInfoCount = static_cast<uint32_t>(infos.size());
    device_cinfo.pQueueCreateInfos = infos.data();
    device_cinfo.pEnabledFeatures = features;
    device_cinfo.enabledExtensionCount = static_cast<uint32_t>(_device_extensions.size());
    device_cinfo.ppEnabledExtensionNames = _device_extensions.data();

    if (_validation_layers_enabled){
        device_cinfo.enabledLayerCount = static_cast<uint32_t>(_validation_layers.size());
        device_cinfo.ppEnabledLayerNames = _validation_layers.data();
    }

    return device_cinfo;
}

VkSwapchainCreateInfoKHR TriangleApplication::swapchain_create_info(VkPhysicalDevice device, 
                                                                    const TriangleApplication::SwapchainSupportDetails& details,
                                                                    VkSurfaceKHR surface) const {
    auto extent = choose_swapchain_extent(details.capabilities);
    auto format = choose_surface_format(details.formats);
    auto mode = choose_present_mode(details.present_mods);

    auto families = find_queue_family(device);
    uint32_t indices[] = {families.graphics_family(), families.present_family()};

    auto imageCount = details.capabilities.minImageCount;
    if (details.capabilities.maxImageCount  > 0 && imageCount > details.capabilities.maxImageCount){
        imageCount = details.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchain_cinfo = {};
    swapchain_cinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_cinfo.imageFormat = format.format;
    swapchain_cinfo.imageColorSpace = format.colorSpace;
    swapchain_cinfo.imageExtent = extent;
    swapchain_cinfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_cinfo.imageArrayLayers = 1;
    swapchain_cinfo.oldSwapchain = VK_NULL_HANDLE;
    swapchain_cinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_cinfo.preTransform = details.capabilities.currentTransform;
    swapchain_cinfo.clipped = VK_TRUE;
    swapchain_cinfo.presentMode = mode;
    swapchain_cinfo.surface = surface;
    swapchain_cinfo.minImageCount = imageCount;

    if (families.graphics_family() != families.present_family()){
        swapchain_cinfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_cinfo.queueFamilyIndexCount = 2;
        swapchain_cinfo.pQueueFamilyIndices = indices;
    } else {
        swapchain_cinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    return swapchain_cinfo;
}

VkImageViewCreateInfo TriangleApplication::image_view_create_info(const VkImage& image) const {
    VkImageViewCreateInfo  view_cinfo = {};
    view_cinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_cinfo.image = image;
    view_cinfo.format = _swapchain_image_format.format;
    view_cinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_cinfo.components.a = view_cinfo.components.r
                            = view_cinfo.components.g
                            = view_cinfo.components.b
                            = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_cinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_cinfo.subresourceRange.baseMipLevel = 0;
    view_cinfo.subresourceRange.levelCount = 1;
    view_cinfo.subresourceRange.baseArrayLayer = 0;
    view_cinfo.subresourceRange.layerCount = 1;

    return view_cinfo;
}

VkShaderModuleCreateInfo TriangleApplication::shader_module_create_info(std::vector<char>& module_data) const {
    VkShaderModuleCreateInfo    module_cinfo = {};
    module_cinfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_cinfo.codeSize = static_cast<uint32_t>(module_data.size());
    module_cinfo.pCode = reinterpret_cast<uint32_t*>(module_data.data());
    return module_cinfo;
}

VkPipelineShaderStageCreateInfo TriangleApplication::pipeline_shader_stage_create_info(const VkShaderModule& module, VkShaderStageFlagBits stage) const {
    VkPipelineShaderStageCreateInfo pipe_cinfo = {};
    pipe_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_cinfo.module = module;
    pipe_cinfo.stage = stage;
    pipe_cinfo.pName = "main";
    return pipe_cinfo;
}

VkRenderPassCreateInfo TriangleApplication::render_pass_create_info(const VkAttachmentDescription& color_desc, const VkSubpassDescription& pass_desc, const std::vector<VkSubpassDependency>& subpass_deps) const {
    VkRenderPassCreateInfo render_pass_cinfo = {};
    render_pass_cinfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_cinfo.attachmentCount = 1;
    render_pass_cinfo.pAttachments = &color_desc;
    render_pass_cinfo.subpassCount = 1;
    render_pass_cinfo.pSubpasses = &pass_desc;
    render_pass_cinfo.dependencyCount = static_cast<uint32_t>(subpass_deps.size());
    render_pass_cinfo.pDependencies = subpass_deps.data();
    return render_pass_cinfo;
}

VkResult TriangleApplication::create_debug_utils_messenger_ext(VkInstance instance, 
                                                               const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
                                                               const VkAllocationCallbacks* p_callbacks,
                                                               VkDebugUtilsMessengerEXT* p_messenger)
{
    constexpr const char* func_name = "vkCreateDebugUtilsMessengerEXT";
    auto create_func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, func_name);
    if (!create_func){
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    return create_func(instance, p_create_info, p_callbacks, p_messenger);
}

VkResult TriangleApplication::destroy_debug_utils_messenger_ext(VkInstance instance,
                                                                VkDebugUtilsMessengerEXT messenger,
                                                                const VkAllocationCallbacks* p_callbacks)
{
    constexpr const char* func_name = "vkDestroyDebugUtilsMessengerEXT";
    auto destroy_func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, func_name);
    if (!destroy_func){
        std::cerr << "Destroy function don't finded\n";
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    destroy_func(instance, messenger, p_callbacks);
    return VK_SUCCESS;
}

void TriangleApplication::populate_required_extensions() {
    uint32_t        extension_count;
    const char**    extension_names;
    extension_names = glfwGetRequiredInstanceExtensions(&extension_count);
    
    for (uint32_t ind = 0; ind < extension_count; ind++){    
        _required_extensions.emplace_back(extension_names[ind]);
    }

    if (_validation_layers_enabled){
        _required_extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
}

void TriangleApplication::populate_validation_layers() {
    _validation_layers = {
        "VK_LAYER_KHRONOS_validation"
    };
}

void TriangleApplication::populate_device_extensions() {
    _device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
}

void TriangleApplication::run() {
    main_loop();
}