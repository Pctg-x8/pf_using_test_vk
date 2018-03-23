#![feature(slice_patterns)]

extern crate appframe;
extern crate ferrite;
extern crate libc;
extern crate pathfinder_font_renderer; use pathfinder_font_renderer::*;
extern crate pathfinder_partitioner;
use pathfinder_partitioner::partitioner::Partitioner;
use pathfinder_partitioner::FillRule;
extern crate pathfinder_path_utils;
use pathfinder_path_utils::cubic_to_quadratic::CubicToQuadraticTransformer;
use pathfinder_path_utils::transform::Transform2DPathIter;
extern crate app_units; use app_units::Au;
extern crate lyon; use lyon::path::builder::{FlatPathBuilder, PathBuilder};
extern crate euclid; use euclid::Transform2D;

use appframe::*;
use ferrite as fe;
use fe::traits::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::borrow::Cow;

const SAMPLE_COUNT: usize = 8;

/*
#[repr(C)] #[derive(Clone)] pub struct Vertex([f32; 4]);
#[repr(C)] #[derive(Clone)] pub struct BufferData { cpuvs: [[f32; 2]; 3], fill_data: [Vertex; 4] }
static BUFFER_DATA: BufferData = BufferData
{
    cpuvs: [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]], fill_data: [
        Vertex([-1.0, -1.0, 0.0, 1.0]), Vertex([1.0, -1.0, 0.0, 1.0]),
        Vertex([-1.0, 1.0, 0.0, 1.0]), Vertex([1.0, 1.0, 0.0, 1.0])
    ]
};
impl BufferData
{
    fn offset_of_fill_data() -> usize
    {
        use std::mem::transmute;
        unsafe { transmute(&transmute::<_, &Self>(0usize).fill_data) }
    }
}*/

/*#[repr(C)] #[derive(Clone, Debug)]
pub struct QBezierSegment([f32; 4], [f32; 4], [f32; 4]);
impl QBezierSegment
{
    pub fn split(&self) -> [QBezierSegment; 2]
    {
        fn center1(a: f32, b: f32) -> f32 { a + (b - a) * 0.5 }
        fn center4(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4]
        {
            [center1(a[0], b[0]), center1(a[1], b[1]), center1(a[2], b[2]), center1(a[3], b[3])]
        }
        let (p0, p1) = (center4(&self.0, &self.1), center4(&self.1, &self.2));
        let new_cp = center4(&p0, &p1);
        [QBezierSegment(self.0.clone(), p0, new_cp.clone()), QBezierSegment(new_cp, p1, self.2.clone())]
    }
}*/

struct ShaderStore
{
    v_curve_pre: fe::ShaderModule, v_pass_fill: fe::ShaderModule,
    f_q_curve: fe::ShaderModule, f_white: fe::ShaderModule
}
impl ShaderStore
{
    fn load(device: &fe::Device) -> Result<Self, Box<std::error::Error>>
    {
        Ok(ShaderStore
        {
            v_curve_pre: fe::ShaderModule::from_file(device, "shaders/curve_pre.vso")?,
            v_pass_fill: fe::ShaderModule::from_file(device, "shaders/pass_fill.vso")?,
            f_q_curve: fe::ShaderModule::from_file(device, "shaders/q_curve.fso")?,
            f_white: fe::ShaderModule::from_file(device, "shaders/white.fso")?
        })
    }
}
const VBIND: &[fe::vk::VkVertexInputBindingDescription] = &[
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 0, stride: std::mem::size_of::<f32>() as u32 * 2, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_VERTEX
    },
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 1, stride: std::mem::size_of::<f32>() as u32 * 4 * 3, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_INSTANCE
    }
];
const VATTRS: &[fe::vk::VkVertexInputAttributeDescription] = &[
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 0, binding: 0, offset: 0, format: fe::vk::VK_FORMAT_R32G32_SFLOAT
    },
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 1, binding: 1, offset: 0, format: fe::vk::VK_FORMAT_R32G32B32A32_SFLOAT
    },
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 2, binding: 1, offset: 16, format: fe::vk::VK_FORMAT_R32G32B32A32_SFLOAT
    },
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 3, binding: 1, offset: 32, format: fe::vk::VK_FORMAT_R32G32B32A32_SFLOAT
    }
];
const VBIND_FILL_PF: &[fe::vk::VkVertexInputBindingDescription] = &[
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 0, stride: std::mem::size_of::<f32>() as u32 * 2, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_VERTEX
    }
];
const VATTRS_FILL_PF: &[fe::vk::VkVertexInputAttributeDescription] = &[
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 0, binding: 0, offset: 0, format: fe::vk::VK_FORMAT_R32G32_SFLOAT
    }
];
/*const VBIND_FILL: &[fe::vk::VkVertexInputBindingDescription] = &[
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 0, stride: std::mem::size_of::<Vertex>() as _, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_VERTEX
    }
];
const VATTRS_FILL: &[fe::vk::VkVertexInputAttributeDescription] = &[
    fe::vk::VkVertexInputAttributeDescription
    {
        binding: 0, location: 0, offset: 0, format: fe::vk::VK_FORMAT_R32G32B32A32_SFLOAT
    }
];*/

struct App
{
    rcmds: RefCell<Option<RenderCommands>>,
    rtdres: RefCell<Option<RenderTargetDependentResources>>,
    rendertargets: RefCell<Option<WindowRenderTargets>>,
    surface: RefCell<Option<fe::Surface>>,
    res: RefCell<Option<Resources>>,
    ferrite: RefCell<Option<Ferrite>>,
    w: RefCell<Option<NativeWindow<App>>>
}
pub struct Ferrite
{
    gq: u32, _tq: u32, queue: fe::Queue, tqueue: fe::Queue, cmdpool: fe::CommandPool, tcmdpool: fe::CommandPool,
    semaphore_sync_next: fe::Semaphore, semaphore_command_completion: fe::Semaphore,
    fence_command_completion: fe::Fence, device_memindex: u32, upload_memindex: u32,
    device: fe::Device, adapter: fe::PhysicalDevice, _d: fe::DebugReportCallback, instance: fe::Instance
}
pub struct Resources
{
    /*buf: fe::Buffer, _dmem: fe::DeviceMemory,*/
    pf_bufs: PathfinderRenderBuffers, pl: fe::PipelineLayout, shaders: ShaderStore
}
pub struct RenderCommands(Vec<fe::CommandBuffer>);
impl App
{
    fn new() -> Self
    {
        App
        {
            w: RefCell::new(None), ferrite: RefCell::new(None),
            rcmds: RefCell::new(None), surface: RefCell::new(None), rendertargets: RefCell::new(None),
            res: RefCell::new(None), rtdres: RefCell::new(None)
        }
    }
}
use pathfinder_partitioner::mesh::Mesh;
use pathfinder_partitioner::BQuadVertexPositions;
pub struct PathfinderRenderBuffers
{
    buf: fe::Buffer, mem: fe::DeviceMemory,
    interior_indices_offset: usize, drawn_vertices: usize
}
impl PathfinderRenderBuffers
{
    pub fn new(f: &Ferrite, mesh: &Mesh) -> fe::Result<Self>
    {
        let pos_buf_size = mesh.b_quad_vertex_positions.len() * std::mem::size_of::<BQuadVertexPositions>();
        let interior_ibuf_size = mesh.b_quad_vertex_interior_indices.len() * std::mem::size_of::<u32>();
        let interior_indices_offset = pos_buf_size;

        let bufsize = pos_buf_size + interior_ibuf_size;
        let buf = fe::BufferDesc::new(bufsize, fe::BufferUsage::VERTEX_BUFFER.index_buffer().transfer_dest()).create(&f.device)?;
        let breq = buf.requirements();
        let mem = fe::DeviceMemory::allocate(&f.device, breq.size as _, f.device_memindex)?;
        buf.bind(&mem, 0)?;
        
        let ubuf = fe::BufferDesc::new(bufsize, fe::BufferUsage::TRANSFER_SRC).create(&f.device)?;
        let ubreq = ubuf.requirements();
        let umem = fe::DeviceMemory::allocate(&f.device, ubreq.size as _, f.upload_memindex)?;
        ubuf.bind(&umem, 0)?;
        unsafe
        {
            let mapped = umem.map(0 .. bufsize)?;
            mapped.slice_mut(0, mesh.b_quad_vertex_positions.len()).clone_from_slice(&mesh.b_quad_vertex_positions);
            mapped.slice_mut(interior_indices_offset, mesh.b_quad_vertex_interior_indices.len())
                .clone_from_slice(&mesh.b_quad_vertex_interior_indices);
        }
        unsafe { umem.unmap(); }
        let init_commands = f.cmdpool.alloc(1, true).unwrap();
        init_commands[0].begin().unwrap()
            .pipeline_barrier(fe::PipelineStageFlags::TOP_OF_PIPE, fe::PipelineStageFlags::TRANSFER, true,
                &[], &[
                    fe::BufferMemoryBarrier::new(&buf, 0 .. bufsize, 0, fe::AccessFlags::TRANSFER.write),
                    fe::BufferMemoryBarrier::new(&ubuf, 0 .. bufsize, 0, fe::AccessFlags::TRANSFER.read)
                ], &[])
            .copy_buffer(&ubuf, &buf, &[fe::vk::VkBufferCopy { srcOffset: 0, dstOffset: 0, size: bufsize as _ }])
            .pipeline_barrier(fe::PipelineStageFlags::TRANSFER, fe::PipelineStageFlags::VERTEX_INPUT, true,
                &[], &[
                    fe::BufferMemoryBarrier::new(&buf, 0 .. bufsize, fe::AccessFlags::TRANSFER.write,
                        fe::AccessFlags::VERTEX_ATTRIBUTE_READ | fe::AccessFlags::INDEX_READ)
                ], &[]);
        f.queue.submit(&[fe::SubmissionBatch
        {
            command_buffers: Cow::Borrowed(&init_commands), .. Default::default()
        }], None)?; f.queue.wait()?;

        return Ok(PathfinderRenderBuffers
        {
            buf, mem, interior_indices_offset,
            drawn_vertices: mesh.b_quad_vertex_interior_indices.len()
        })
    }
}
impl EventDelegate for App
{
    fn postinit(&self, server: &Rc<GUIApplication<Self>>)
    {
        extern "system" fn dbg_cb(_flags: fe::vk::VkDebugReportFlagsEXT, _object_type: fe::vk::VkDebugReportObjectTypeEXT,
            _object: u64, _location: libc::size_t, _message_code: i32, _layer_prefix: *const libc::c_char,
            message: *const libc::c_char, _user_data: *mut libc::c_void) -> fe::vk::VkBool32
        {
            println!("dbg_cb: {}", unsafe { std::ffi::CStr::from_ptr(message).to_str().unwrap() });
            false as _
        }

        #[cfg(target_os = "macos")] const PLATFORM_SURFACE: &str = "VK_MVK_macos_surface";
        #[cfg(windows)] const PLATFORM_SURFACE: &str = "VK_KHR_win32_surface";
        #[cfg(feature = "with_xcb")] const PLATFORM_SURFACE: &str = "VK_KHR_xcb_surface";
        let instance = fe::InstanceBuilder::new("appframe_integ", (0, 1, 0), "Ferrite", (0, 1, 0))
            .add_extensions(vec!["VK_KHR_surface", PLATFORM_SURFACE, "VK_EXT_debug_report"])
            .add_layer("VK_LAYER_LUNARG_standard_validation")
            .create().unwrap();
        let d = fe::DebugReportCallbackBuilder::new(&instance, dbg_cb).report_error().report_warning()
            .report_performance_warning().create().unwrap();
        let adapter = instance.iter_physical_devices().unwrap().next().unwrap();
        println!("Vulkan AdapterName: {}", unsafe { std::ffi::CStr::from_ptr(adapter.properties().deviceName.as_ptr()).to_str().unwrap() });
        let memindices = adapter.memory_properties();
        let qfp = adapter.queue_family_properties();
        let gq = qfp.find_matching_index(fe::QueueFlags::GRAPHICS).expect("Cannot find a graphics queue");
        let tq = qfp.find_another_matching_index(fe::QueueFlags::TRANSFER, gq)
            .or_else(|| qfp.find_matching_index(fe::QueueFlags::TRANSFER)).expect("No transferrable queue family found");
        let united_queue = gq == tq;
        let qs = if united_queue { vec![fe::DeviceQueueCreateInfo(gq, vec![0.0; 2])] }
            else { vec![fe::DeviceQueueCreateInfo(gq, vec![0.0]), fe::DeviceQueueCreateInfo(tq, vec![0.0])] };
        let device = fe::DeviceBuilder::new(&adapter)
            .add_extensions(vec!["VK_KHR_swapchain"]).add_queues(qs)
            .enable_fill_mode_nonsolid().enable_sample_rate_shading()
            .create().unwrap();
        let queue = device.queue(gq, 0);
        let cmdpool = fe::CommandPool::new(&device, gq, false, false).unwrap();
        let device_memindex = memindices.find_device_local_index().unwrap();
        let upload_memindex = memindices.find_host_visible_index().unwrap();
        let ferrite = Ferrite
        {
            fence_command_completion: fe::Fence::new(&device, false).unwrap(),
            semaphore_sync_next: fe::Semaphore::new(&device).unwrap(),
            semaphore_command_completion: fe::Semaphore::new(&device).unwrap(),
            tcmdpool: fe::CommandPool::new(&device, tq, false, false).unwrap(),
            cmdpool, queue, tqueue: device.queue(tq, if united_queue { 1 } else { 0 }),
            device, adapter, instance, gq, _tq: tq, _d: d, device_memindex, upload_memindex
        };

        let mut fc = FontContext::new().unwrap();
        let open_sans_regular = std::fs::File::open("OpenSans-Regular.ttf")
            .and_then(|mut fp| { use std::io::prelude::*; let mut b = Vec::new(); fp.read_to_end(&mut b).map(|_| b) })
            .unwrap().into();
        fc.add_font_from_memory(&0, open_sans_regular, 0).unwrap();
        // FontInstanceのサイズ指定はデバイス依存px単位
        let font = FontInstance::new(&0, Au::from_f32_px(80.0 * 96.0 / 72.0));
        // text -> glyph indices
        let glyphs = fc.load_glyph_indices_for_characters(&font, &"Hello, world!".chars().map(|x| x as _).collect::<Vec<_>>()).unwrap();
        // glyph indices -> layouted text outlines
        let mut paths = Vec::new();
        let (mut left_offs, mut max_height) = (0.0, 0.0f32);
        for g in glyphs
        {
            let mut g = GlyphKey::new(g as _, SubpixelOffset(0));
            let dim = fc.glyph_dimensions(&font, &g, false).unwrap();
            println!("dimension?: {:?}", dim);
            let rendered: f32 = left_offs;
            g.subpixel_offset.0 = (rendered.fract() * SUBPIXEL_GRANULARITY as f32) as _;
            let outline = fc.glyph_outline(&font, &g).unwrap();
            paths.extend(Transform2DPathIter::new(outline.iter(),
                &Transform2D::create_translation(rendered.trunc() as _, 0.0)));
            left_offs += dim.advance/* * 60.0*/;
            max_height = max_height.max(dim.size.height as f32/* as i32 as f32 / 96.0*/);
        }
        println!("left offset: {}", left_offs);
        println!("max height: {}", max_height);
        // text outlines -> pathfinder mesh
        let mut partitioner = Partitioner::new();
        for pe in Transform2DPathIter::new(paths.iter().cloned(), &Transform2D::create_translation(-left_offs * 0.5, -max_height * 0.5))
        {
            partitioner.builder_mut().path_event(pe);
        }
        partitioner.partition(FillRule::Winding);
        partitioner.builder_mut().build_and_reset();
        partitioner.mesh_mut().push_stencil_segments(CubicToQuadraticTransformer::new(paths.iter().cloned(), 5.0));
        partitioner.mesh_mut().push_stencil_normals(CubicToQuadraticTransformer::new(paths.iter().cloned(), 5.0));
        let mesh = partitioner.into_mesh();
        // println!("debug: b_quad_vertex_interior_indices: {:?}", mesh.b_quad_vertex_positions);
        // println!("debug: b_quad_vertex_interior_indices: {:?}", mesh.b_quad_vertex_interior_indices);

        /*let qb1 = QBezierSegment([-1.0, 1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]);
        let [qb2, qb3] = qb1.split();
        let ([qb4, qb5], [qb6, qb7]) = (qb2.split(), qb3.split());
        let controlpoints_size = std::mem::size_of::<QBezierSegment>() * 4;*/
        
        // let total_bufsize = std::mem::size_of::<BufferData>() + controlpoints_size;
        /*let total_bufsize = mesh.b_quad_vertex_positions.len() * 2 * std::mem::size_of::<f32>() * 6
            + mesh.b_quad_vertex_interior_indices.len() * std::mem::size_of::<u32>();
        let buf = fe::BufferDesc::new(total_bufsize, fe::BufferUsage::VERTEX_BUFFER.transfer_dest()).create(&device).unwrap();
        let memreq = buf.requirements();
        let dmem = fe::DeviceMemory::allocate(&device, memreq.size as _, device_memindex).unwrap();
        {
            let upload_buf = fe::BufferDesc::new(total_bufsize, fe::BufferUsage::VERTEX_BUFFER.transfer_src()).create(&device).unwrap();
            let upload_memreq = upload_buf.requirements();
            let upload_mem = fe::DeviceMemory::allocate(&device, upload_memreq.size as _, upload_memindex).unwrap();
            buf.bind(&dmem, 0).unwrap(); upload_buf.bind(&upload_mem, 0).unwrap();
            unsafe 
            {
                let mapped = upload_mem.map(0 .. total_bufsize).unwrap();
                *mapped.get_mut(0) = BUFFER_DATA.clone();
                mapped.slice_mut(std::mem::size_of::<BufferData>(), 4).clone_from_slice(&[qb4, qb5, qb6, qb7]);
            }
            let fw = fe::Fence::new(&device, false).unwrap();
            let init_commands = cmdpool.alloc(1, true).unwrap();
            init_commands[0].begin().unwrap()
                .pipeline_barrier(fe::PipelineStageFlags::TOP_OF_PIPE, fe::PipelineStageFlags::TRANSFER, true,
                    &[], &[
                        fe::BufferMemoryBarrier::new(&buf, 0 .. total_bufsize, 0, fe::AccessFlags::TRANSFER.write),
                        fe::BufferMemoryBarrier::new(&upload_buf, 0 .. total_bufsize, 0, fe::AccessFlags::TRANSFER.read)
                    ], &[])
                .copy_buffer(&upload_buf, &buf, &[fe::vk::VkBufferCopy { srcOffset: 0, dstOffset: 0, size: total_bufsize as _ }])
                .pipeline_barrier(fe::PipelineStageFlags::TRANSFER, fe::PipelineStageFlags::VERTEX_INPUT, true,
                    &[], &[
                        fe::BufferMemoryBarrier::new(&buf, 0 .. total_bufsize, fe::AccessFlags::TRANSFER.write,
                            fe::AccessFlags::VERTEX_ATTRIBUTE_READ)
                    ], &[]);
            queue.submit(&[fe::SubmissionBatch
            {
                command_buffers: Cow::Borrowed(&init_commands), .. Default::default()
            }], Some(&fw)).unwrap(); fw.wait().unwrap();
        }*/
        *self.res.borrow_mut() = Some(Resources
        {
            pf_bufs: PathfinderRenderBuffers::new(&ferrite, &mesh).unwrap(),
            shaders: ShaderStore::load(&ferrite.device).unwrap(),
            pl: fe::PipelineLayout::new(&ferrite.device, &[], &[(fe::ShaderStage::VERTEX, 0 .. 8)]).unwrap()
        });
        *self.ferrite.borrow_mut() = Some(ferrite);

        let w = NativeWindowBuilder::new(640, 360, "Loop-Blinn Bezier Curve Drawing Example").create_renderable(server).unwrap();
        *self.w.borrow_mut() = Some(w);
        self.w.borrow().as_ref().unwrap().show();
    }
    fn on_init_view(&self, server: &GUIApplication<Self>, surface_onto: &NativeView<Self>)
    {
        let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();

        if !server.presentation_support(&f.adapter, f.gq) { panic!("Vulkan Rendering is not supported by platform"); }
        let surface = server.create_surface(surface_onto, &f.instance).unwrap();
        if !f.adapter.surface_support(f.gq, &surface).unwrap() { panic!("Vulkan Rendering is not supported to this surface"); }
        *self.surface.borrow_mut() = Some(surface);
        let rtvs = self.init_swapchains().unwrap().unwrap();
        *self.rtdres.borrow_mut() = Some(RenderTargetDependentResources::new(&f.device,
            self.res.borrow().as_ref().unwrap(), &rtvs).unwrap());
        *self.rendertargets.borrow_mut() = Some(rtvs);
        *self.rcmds.borrow_mut() = Some(self.populate_render_commands().unwrap());
    }
    fn on_render_period(&self)
    {
        if self.ensure_render_targets().unwrap()
        {
            if let Err(e) = self.render()
            {
                if e.0 == fe::vk::VK_ERROR_OUT_OF_DATE_KHR
                {
                    // Require to recreate resources(discarding resources)
                    let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();

                    f.fence_command_completion.wait().unwrap(); f.fence_command_completion.reset().unwrap();
                    *self.rcmds.borrow_mut() = None;
                    *self.rtdres.borrow_mut() = None;
                    *self.rendertargets.borrow_mut() = None;

                    // reissue rendering
                    self.on_render_period();
                }
                else { let e: fe::Result<()> = Err(e); e.unwrap(); }
            }
        }
    }
}
impl App
{
    fn ensure_render_targets(&self) -> fe::Result<bool>
    {
        if self.rendertargets.borrow().is_none()
        {
            let rtv = self.init_swapchains()?;
            if rtv.is_none() { return Ok(false); }
            *self.rendertargets.borrow_mut() = rtv;
        }
        if self.rtdres.borrow().is_none()
        {
            let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();
            let resr = self.res.borrow(); let res = resr.as_ref().unwrap();
            *self.rtdres.borrow_mut() = Some(RenderTargetDependentResources::new(&f.device, res,
                self.rendertargets.borrow().as_ref().unwrap())?);
        }
        if self.rcmds.borrow().is_none()
        {
            *self.rcmds.borrow_mut() = Some(self.populate_render_commands().unwrap());
        }
        Ok(true)
    }
    fn init_swapchains(&self) -> fe::Result<Option<WindowRenderTargets>>
    {
        let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();
        let sr = self.surface.borrow(); let s = sr.as_ref().unwrap();

        let surface_caps = f.adapter.surface_capabilities(s)?;
        let surface_format = f.adapter.surface_formats(s)?.into_iter()
            .find(|f| fe::FormatQuery(f.format).eq_bit_width(32).is_component_of(fe::FormatComponents::RGBA).has_element_of(fe::ElementType::UNORM).passed()).unwrap();
        let surface_pm = f.adapter.surface_present_modes(s)?.remove(0);
        let surface_ca = if (surface_caps.supportedCompositeAlpha & fe::CompositeAlpha::PostMultiplied as u32) != 0
        {
            fe::CompositeAlpha::PostMultiplied
        }
        else { fe::CompositeAlpha::Opaque };
        let surface_size = match surface_caps.currentExtent
        {
            fe::vk::VkExtent2D { width: 0xffff_ffff, height: 0xffff_ffff } => fe::Extent2D(640, 360),
            fe::vk::VkExtent2D { width, height } => fe::Extent2D(width, height)
        };
        if surface_size.0 <= 0 || surface_size.1 <= 0 { return Ok(None); }
        let swapchain = fe::SwapchainBuilder::new(s, surface_caps.minImageCount.max(2),
            &surface_format, &surface_size, fe::ImageUsage::COLOR_ATTACHMENT)
                .present_mode(surface_pm).pre_transform(fe::SurfaceTransform::Identity)
                .composite_alpha(surface_ca).create(&f.device)?;
        // acquire_nextより前にやらないと死ぬ(get_images)
        let backbuffers = swapchain.get_images()?;
        let isr = fe::ImageSubresourceRange::color(0, 0);
        let bb_views = backbuffers.iter().map(|i| i.create_view(None, None, &fe::ComponentMapping::default(), &isr))
            .collect::<fe::Result<Vec<_>>>()?;
        
        let ms_dest = fe::ImageDesc::new(&surface_size, surface_format.format, fe::ImageUsage::COLOR_ATTACHMENT.transient_attachment(),
            fe::ImageLayout::Undefined).sample_counts(SAMPLE_COUNT as _).create(&f.device)
                .and_then(|r| ImageMemoryPair::new(r, f.device_memindex)).unwrap();

        let mut rpb = fe::RenderPassBuilder::new();
        rpb.add_attachments(vec![
            fe::AttachmentDescription::new(surface_format.format, fe::ImageLayout::ColorAttachmentOpt, fe::ImageLayout::ColorAttachmentOpt)
                .load_op(fe::LoadOp::Clear).samples(SAMPLE_COUNT as _),
            fe::AttachmentDescription::new(surface_format.format, fe::ImageLayout::PresentSrc, fe::ImageLayout::PresentSrc)
                .store_op(fe::StoreOp::Store)
        ]);
        rpb.add_subpass(fe::SubpassDescription::new()
            .add_color_output(0, fe::ImageLayout::ColorAttachmentOpt, Some((1, fe::ImageLayout::ColorAttachmentOpt))));
        rpb.add_dependency(fe::vk::VkSubpassDependency
        {
            srcSubpass: fe::vk::VK_SUBPASS_EXTERNAL, dstSubpass: 0,
            srcStageMask: fe::PipelineStageFlags::TOP_OF_PIPE.0, dstStageMask: fe::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.0,
            srcAccessMask: fe::AccessFlags::MEMORY.read, dstAccessMask: fe::AccessFlags::COLOR_ATTACHMENT.write,
            dependencyFlags: fe::vk::VK_DEPENDENCY_BY_REGION_BIT
        });
        let rp = rpb.create(&f.device).unwrap();
        let framebuffers = bb_views.iter().map(|iv| fe::Framebuffer::new(&rp, &[&ms_dest.view, iv], &surface_size, 1))
            .collect::<fe::Result<Vec<_>>>()?;

        let fw = fe::Fence::new(&f.device, false).unwrap();
        let init_commands = f.cmdpool.alloc(1, true).unwrap();
        let membarriers = bb_views.iter().map(|iv| fe::ImageMemoryBarrier::new(&fe::ImageSubref::color(&iv, 0, 0),
            fe::ImageLayout::Undefined, fe::ImageLayout::PresentSrc))
            .chain(vec![fe::ImageMemoryBarrier::new(&fe::ImageSubref::color(&ms_dest.view, 0, 0),
                fe::ImageLayout::Undefined, fe::ImageLayout::ColorAttachmentOpt)]).collect::<Vec<_>>();
        init_commands[0].begin().unwrap()
            .pipeline_barrier(fe::PipelineStageFlags::TOP_OF_PIPE, fe::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                true, &[], &[], &membarriers);
        f.queue.submit(&[fe::SubmissionBatch
        {
            command_buffers: Cow::Borrowed(&init_commands), .. Default::default()
        }], Some(&fw)).unwrap(); fw.wait().unwrap();
        
        Ok(Some(WindowRenderTargets
        {
            swapchain, backbuffers: bb_views, framebuffers, renderpass: rp, size: surface_size, ms_dest
        }))
    }
    fn populate_render_commands(&self) -> fe::Result<RenderCommands>
    {
        let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();
        let rtvr = self.rendertargets.borrow(); let rtvs = rtvr.as_ref().unwrap();
        let resr = self.res.borrow(); let res = resr.as_ref().unwrap();
        let rdsr = self.rtdres.borrow(); let rds = rdsr.as_ref().unwrap();

        let render_commands = f.cmdpool.alloc(rtvs.framebuffers.len() as _, true)?;
        for (c, fb) in render_commands.iter().zip(&rtvs.framebuffers)
        {
            c.begin()?
                .begin_render_pass(&rtvs.renderpass, fb, fe::vk::VkRect2D
                {
                    offset: fe::vk::VkOffset2D { x: 0, y: 0 },
                    extent: fe::vk::VkExtent2D { width: rtvs.size.0, height: rtvs.size.1 }
                }, &[fe::ClearValue::Color([0.0, 0.0, 0.0, 1.0])], true)
                    .bind_graphics_pipeline_pair(&rds.gp_simple_fill, &res.pl)
                    .bind_vertex_buffers(0, &[(&res.pf_bufs.buf, 0)])
                    .bind_index_buffer(&res.pf_bufs.buf, res.pf_bufs.interior_indices_offset, fe::IndexType::U32)
                    .push_graphics_constant(fe::ShaderStage::VERTEX, 0, &[rtvs.size.0 as f32, rtvs.size.1 as _])
                    .draw_indexed(res.pf_bufs.drawn_vertices as _, 1, 0, 0, 0)
                    // .bind_graphics_pipeline(&rds.gp_wire).draw(VERTEX_DATA.len() as _, 1, 0, 0)
                .end_render_pass();
        }
        Ok(RenderCommands(render_commands))
    }
    fn render(&self) -> fe::Result<()>
    {
        let fr = self.ferrite.borrow(); let f = fr.as_ref().unwrap();
        let rtvr = self.rendertargets.borrow(); let rtvs = rtvr.as_ref().unwrap();
        let rcmdsr = self.rcmds.borrow(); let rcmds = rcmdsr.as_ref().unwrap();

        let next = rtvs.swapchain.acquire_next(None, fe::CompletionHandler::Device(&f.semaphore_sync_next))?
            as usize;
        f.queue.submit(&[fe::SubmissionBatch
        {
            command_buffers: Cow::Borrowed(&rcmds.0[next..next+1]),
            wait_semaphores: Cow::Borrowed(&[(&f.semaphore_sync_next, fe::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)]),
            signal_semaphores: Cow::Borrowed(&[&f.semaphore_command_completion])
        }], Some(&f.fence_command_completion))?;
        f.queue.present(&[(&rtvs.swapchain, next as _)], &[&f.semaphore_command_completion])?;
        // コマンドバッファの使用が終了したことを明示する
        f.fence_command_completion.wait()?; f.fence_command_completion.reset()?; Ok(())
    }
}
#[allow(dead_code)]
struct WindowRenderTargets
{
    framebuffers: Vec<fe::Framebuffer>, renderpass: fe::RenderPass,
    ms_dest: ImageMemoryPair, backbuffers: Vec<fe::ImageView>,
    swapchain: fe::Swapchain, size: fe::Extent2D
}
struct RenderTargetDependentResources
{
    gp_fill: fe::Pipeline, gp_simple_fill: fe::Pipeline, gp_wire: fe::Pipeline
}
impl RenderTargetDependentResources
{
    pub fn new(device: &fe::Device, res: &Resources, rtvs: &WindowRenderTargets) -> fe::Result<Self>
    {
        let vp = fe::vk::VkViewport
        {
            x: 0.0, y: 0.0, width: rtvs.size.0 as _, height: rtvs.size.1 as _, minDepth: 0.0, maxDepth: 1.0
        };
        let scis = fe::vk::VkRect2D
        {
            offset: fe::vk::VkOffset2D { x: 0, y: 0 },
            extent: fe::vk::VkExtent2D { width: vp.width as _, height: vp.height as _ }
        };
        let mut vps = fe::VertexProcessingStages::new(fe::PipelineShader::new(&res.shaders.v_curve_pre, "main", None),
            VBIND, VATTRS, fe::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vps.fragment_shader(fe::PipelineShader::new(&res.shaders.f_q_curve, "main", None));
        let mut ms = fe::MultisampleState::new();
        ms.rasterization_samples(SAMPLE_COUNT).sample_shading(Some(1.0)).sample_mask(&[(1 << SAMPLE_COUNT) - 1]);
        let mut gpb = fe::GraphicsPipelineBuilder::new(&res.pl, (&rtvs.renderpass, 0));
        gpb.vertex_processing(vps)
            .fixed_viewport_scissors(fe::DynamicArrayState::Static(&[vp]), fe::DynamicArrayState::Static(&[scis]))
            .add_attachment_blend(fe::AttachmentColorBlendState::premultiplied());
        gpb.multisample_state(Some(&ms));
        let gp_fill = gpb.create(device, None)?;
        let mut vps_simple = fe::VertexProcessingStages::new(fe::PipelineShader::new(&res.shaders.v_pass_fill, "main", None),
            VBIND_FILL_PF, VATTRS_FILL_PF, fe::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vps_simple.fragment_shader(fe::PipelineShader::new(&res.shaders.f_white, "main", None));
        gpb.vertex_processing(vps_simple);
        let gp_simple_fill = gpb.create(device, None)?;
        gpb.vertex_processing_mut().fragment_shader(fe::PipelineShader::new(&res.shaders.f_white, "main", None));
        gpb.polygon_mode(fe::vk::VK_POLYGON_MODE_LINE);
        let gp_wire = gpb.create(device, None)?;
        
        Ok(RenderTargetDependentResources { gp_fill, gp_simple_fill, gp_wire })
    }
}

#[allow(dead_code)]
struct ImageMemoryPair { view: fe::ImageView, image: fe::Image, memory: fe::DeviceMemory }
impl ImageMemoryPair
{
    fn new(image: fe::Image, memory_index: u32) -> fe::Result<Self>
    {
        let ireq = image.requirements();
        let memory = fe::DeviceMemory::allocate(image.device(), ireq.size as _, memory_index)?;
        image.bind(&memory, 0)?;
        let view = image.create_view(None, None, &fe::ComponentMapping::default(), &fe::ImageSubresourceRange::color(0..1, 0..1))?;
        return Ok(ImageMemoryPair { view, image, memory })
    }
}

fn main() { std::process::exit(GUIApplication::run(App::new())); }
