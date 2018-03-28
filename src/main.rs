#![feature(slice_patterns)]

extern crate appframe;
extern crate ferrite;
extern crate libc;
extern crate pathfinder_font_renderer; use pathfinder_font_renderer::*;
extern crate pathfinder_partitioner;
use pathfinder_partitioner::partitioner::Partitioner;
use pathfinder_partitioner::FillRule;
use pathfinder_partitioner::BVertexLoopBlinnData;
extern crate pathfinder_path_utils;
// use pathfinder_path_utils::cubic_to_quadratic::CubicToQuadraticTransformer;
use pathfinder_path_utils::transform::Transform2DPathIter;
extern crate app_units; use app_units::Au;
extern crate lyon; use lyon::path::builder::{FlatPathBuilder, PathBuilder};
extern crate euclid; use euclid::Transform2D;

use appframe::*;
use ferrite as fe;
use fe::traits::*;
use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::borrow::Cow;

const SAMPLE_COUNT: usize = 8;

struct ShaderStore
{
    v_curve_pre: fe::ShaderModule, v_pass_fill: fe::ShaderModule,
    f_white_curve: fe::ShaderModule, f_white: fe::ShaderModule
}
impl ShaderStore
{
    fn load(device: &fe::Device) -> Result<Self, Box<std::error::Error>>
    {
        Ok(ShaderStore
        {
            v_curve_pre: fe::ShaderModule::from_file(device, "shaders/curve_pre.vso")?,
            v_pass_fill: fe::ShaderModule::from_file(device, "shaders/pass_fill.vso")?,
            f_white_curve: fe::ShaderModule::from_file(device, "shaders/white_curve.fso")?,
            f_white: fe::ShaderModule::from_file(device, "shaders/white.fso")?
        })
    }
}
const VBIND_LB: &[fe::vk::VkVertexInputBindingDescription] = &[
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 0, stride: std::mem::size_of::<f32>() as u32 * 2, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_VERTEX
    },
    fe::vk::VkVertexInputBindingDescription
    {
        binding: 1, stride: std::mem::size_of::<BVertexLoopBlinnData>() as _, inputRate: fe::vk::VK_VERTEX_INPUT_RATE_VERTEX
    }
];
const VATTRS_LB: &[fe::vk::VkVertexInputAttributeDescription] = &[
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 0, binding: 0, offset: 0, format: fe::vk::VK_FORMAT_R32G32_SFLOAT
    },
    fe::vk::VkVertexInputAttributeDescription
    {
        location: 1, binding: 1, offset: 0, format: fe::vk::VK_FORMAT_R8G8B8A8_SINT
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

// Pathfinder Rendering PushConstants //
const PF_DIRECT_RENDER_RT_PIXELS_OFFSET: u32 = 0;
const PF_DIRECT_RENDER_GLYPH_INDEX_OFFSET: u32 = 4 * 2;
const PF_DIRECT_RENDER_PUSH_CONSTANT_LAYOUT: &[(fe::ShaderStage, Range<u32>)] = &[
    (fe::ShaderStage::VERTEX, PF_DIRECT_RENDER_RT_PIXELS_OFFSET .. PF_DIRECT_RENDER_RT_PIXELS_OFFSET + 4 * 3)
];

struct App
{
    rcmds: RefCell<Option<RenderCommands>>,
    rendertargets: RefCell<Option<WindowRenderTargets>>,

    surface: RefCell<Option<(fe::Surface, SurfaceDesc)>>,
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
impl App
{
    fn new() -> Self
    {
        App
        {
            w: RefCell::new(None), ferrite: RefCell::new(None),
            rcmds: RefCell::new(None), surface: RefCell::new(None), rendertargets: RefCell::new(None),
            res: RefCell::new(None)
        }
    }
}
use pathfinder_partitioner::mesh::Mesh;
use pathfinder_partitioner::BQuadVertexPositions;
use std::ops::Range;
#[repr(C)] #[derive(Clone, Debug)]
pub struct Hint { x_height: f32, hinted_x_height: f32, stem_height: f32, hinted_stem_height: f32 }
impl Hint
{
    fn compute(fc: &mut FontContext<usize>, instance: &FontInstance<usize>, use_hinting: bool) -> Self
    {
        let ppu = fc.pixels_per_unit(instance).unwrap();
        let x_height = fc.x_height(instance).unwrap() as f32;
        let stem_height = fc.cap_height(instance).unwrap() as f32;
        let (hinted_x_height, hinted_stem_height);
        if use_hinting
        {
            hinted_x_height = x_height;
            hinted_stem_height = stem_height;
        }
        else
        {
            hinted_x_height = f32::round(f32::round(x_height as f32 * ppu) / ppu);
            hinted_stem_height = f32::round(f32::round(stem_height as f32 * ppu) / ppu);
        }

        Hint { x_height, stem_height, hinted_x_height, hinted_stem_height }
    }
}
#[repr(C)] #[derive(Clone, Debug)]
pub struct GlyphTransform { st: [f32; 4], ext: [f32; 2], pad: [f32; 2] }

pub struct BufferPrealloc { total: usize }
impl BufferPrealloc
{
    pub fn new() -> Self { BufferPrealloc { total: 0 } }
    pub fn alloc(&mut self, size: usize, align: usize) -> Range<usize>
    {
        let align = align.max(1);
        let o = self.total + (align - 1) & !(align - 1);
        self.total = o + size; return o .. self.total;
    }
}
/*
Buffer:
[[Interior-Filling Positions(Vertex Buffer)]] - 0..
[[Interior-Filling Indices(Index Buffer)]]    - interior_indices_offset..
[[Loop-Blinn Curve Data(Vertex Buffer)]]      - bvlb_data_offset..
[[Loop-Blinn Curve Positions(Vertex Buffer)]] - bvlb_pos_offset..
[[Loop-Blinn Curve Indices(Index Buffer)]]    - bvlb_indices_offset..
[[Hint Constant(Uniform Buffer)]]
[[Glyph Transform Constants(Uniform Texel Buffer)]]
*/
struct CurveRenderDataRanges { data: Range<usize>, pos: Range<usize>, indices: Range<usize> }
#[allow(dead_code)]
pub struct PathfinderRenderBuffers
{
    buf: fe::Buffer, mem: fe::DeviceMemory,
    descset: fe::vk::VkDescriptorSet, transforms_view: fe::BufferView, descpool: fe::DescriptorPool,
    interior_indices_range: Range<usize>, interior_pos_range: Range<usize>, interior_vertex_range_per_mesh: Vec<Range<usize>>,
    curve_render_data_ranges: Option<CurveRenderDataRanges>, bvlb_vertex_range_per_mesh: Vec<Range<usize>>
}
impl PathfinderRenderBuffers
{
    pub fn new(f: &Ferrite, utb_desc_layout: &fe::DescriptorSetLayout,
        mesh: Vec<Mesh>, transforms: Vec<GlyphTransform>, hint: Hint) -> fe::Result<Self>
    {
        use std::mem::size_of;

        let mut prealloc = BufferPrealloc::new();
        let min_uniform_align = f.adapter.properties().limits.minUniformBufferOffsetAlignment as usize;
        let hint_const_range = prealloc.alloc(size_of::<Hint>(), min_uniform_align);
        let transforms_range = prealloc.alloc(size_of::<GlyphTransform>() * transforms.len(), min_uniform_align);
        let interior_pos_range = prealloc.alloc(mesh.iter().fold(0, |a, x| a + x.b_quad_vertex_positions.len()) * size_of::<BQuadVertexPositions>(), 1);
        let interior_indices_range = prealloc.alloc(mesh.iter().fold(0, |a, x| a + x.b_quad_vertex_interior_indices.len()) * size_of::<u32>(), 1);
        /*let vertex_positions = mesh.iter().map(|x| x.b_quad_vertex_positions.len()).fold(0, |a, b| a + b);
        let pos_buf_size = vertex_positions * size_of::<BQuadVertexPositions>();
        let interior_indices = mesh.iter().map(|x| x.b_quad_vertex_interior_indices.len()).fold(0, |a, b| a + b);
        let interior_ibuf_size = interior_indices * size_of::<u32>();*/
        // println!("counter: {} {}", vertex_positions, interior_indices);
        // let interior_indices_offset = 0 + pos_buf_size;
        /*let bvlb_data_size = mesh.iter().map(|x| x.b_vertex_loop_blinn_data.len()).fold(0, |a, b| a + b)
            * size_of::<BVertexLoopBlinnData>();
        let bvlb_data_offset = interior_indices_offset + interior_ibuf_size;
        let bvlb_pos_size = mesh.iter().map(|x| x.b_vertex_positions.len()).fold(0, |a, b| a + b) * size_of::<f32>() * 2;
        let bvlb_pos_offset = bvlb_data_offset + bvlb_data_size;
        let bvlb_indices_offset = bvlb_pos_offset + bvlb_pos_size;*/
        let mut bvlb_indices = Vec::new();
        let mut bvlb_vertex_range_per_mesh: Vec<Range<usize>> = Vec::with_capacity(mesh.len());
        let mut interior_vertex_range_per_mesh: Vec<Range<usize>> = Vec::with_capacity(mesh.len());
        let mut bvlb_start = 0;
        for m in &mesh
        {
            let last_iv_end = interior_vertex_range_per_mesh.last().map(|x| x.end).unwrap_or(0);
            let iv_range = last_iv_end .. last_iv_end + m.b_quad_vertex_interior_indices.len();
            // println!("interior: {:?}", iv_range);
            interior_vertex_range_per_mesh.push(iv_range);
            let mut curve_index_count = 0;
            for bq in &m.b_quads
            {
                // active curve(upper)
                if bq.upper_control_point_vertex_index != 0xffff_ffff
                {
                    bvlb_indices.extend(vec![bvlb_start + bq.upper_control_point_vertex_index,
                        bvlb_start + bq.upper_right_vertex_index, bvlb_start + bq.upper_left_vertex_index]);
                    curve_index_count += 3;
                }
                // active curve(lower)
                if bq.lower_control_point_vertex_index != 0xffff_ffff
                {
                    bvlb_indices.extend(vec![bvlb_start + bq.lower_control_point_vertex_index,
                        bvlb_start + bq.lower_right_vertex_index, bvlb_start + bq.lower_left_vertex_index]);
                    curve_index_count += 3;
                }
            }
            // println!("bvlb: {} {}", m.b_vertex_loop_blinn_data.len(), m.b_vertex_positions.len());
            let last_bv_end = bvlb_vertex_range_per_mesh.last().map(|x| x.end).unwrap_or(0);
            let bv_range = last_bv_end .. last_bv_end + curve_index_count;
            // println!("curve: {:?}", bv_range);
            bvlb_vertex_range_per_mesh.push(bv_range);
            bvlb_start += m.b_vertex_loop_blinn_data.len() as u32;
        }
        let curve_render_data_ranges = if bvlb_indices.is_empty() { None } else
        {
            let data = prealloc.alloc(mesh.iter().fold(0, |a, x| a + x.b_vertex_loop_blinn_data.len()) * size_of::<BVertexLoopBlinnData>(), 1);
            let pos = prealloc.alloc(mesh.iter().fold(0, |a, x| a + x.b_vertex_positions.len()) * size_of::<f32>() * 2, 1);
            let indices = prealloc.alloc(bvlb_indices.len() * size_of::<u32>(), 1);
            Some(CurveRenderDataRanges { data, pos, indices })
        };

        println!("Allocating DeviceMemory: {} bytes", prealloc.total);
        let buf = fe::BufferDesc::new(prealloc.total, fe::BufferUsage::VERTEX_BUFFER.index_buffer()
            .uniform_texel_buffer().uniform_buffer().transfer_dest()).create(&f.device)?;
        let breq = buf.requirements();
        let mem = fe::DeviceMemory::allocate(&f.device, breq.size as _, f.device_memindex)?;
        buf.bind(&mem, 0)?;
        
        let ubuf = fe::BufferDesc::new(prealloc.total, fe::BufferUsage::TRANSFER_SRC).create(&f.device)?;
        let ubreq = ubuf.requirements();
        let umem = fe::DeviceMemory::allocate(&f.device, ubreq.size as _, f.upload_memindex)?;
        ubuf.bind(&umem, 0)?;
        unsafe
        {
            let mapped = umem.map(0 .. prealloc.total)?;
            let (mut vp_offs, mut ii_offs, mut cd_offs, mut cp_offs) = (0, 0, 0, 0);
            for m in &mesh
            {
                // println!("Copying Position: {}({}) ..> {}", vp_offs * size_of::<BQuadVertexPositions>(), vp_offs, m.b_quad_vertex_positions.len());
                mapped.slice_mut(vp_offs * size_of::<BQuadVertexPositions>() + interior_pos_range.start, m.b_quad_vertex_positions.len())
                    .clone_from_slice(&m.b_quad_vertex_positions);
                // println!("Copying Interior Indices: {}({}) ..> {} (+{})", ii_offs * size_of::<u32>(), ii_offs, m.b_quad_vertex_interior_indices.len(), vp_offs);
                // println!("{:?} {:?}", &m.b_quad_vertex_interior_indices[..6],
                //     &m.b_quad_vertex_interior_indices[m.b_quad_vertex_interior_indices.len() - 6..]);
                let s = mapped.slice_mut(ii_offs * size_of::<u32>() + interior_indices_range.start, m.b_quad_vertex_interior_indices.len());
                for (s, d) in m.b_quad_vertex_interior_indices.iter().zip(s.iter_mut())
                {
                    // println!("Writing {}", s + vp_offs as u32);
                    *d = s + 6 * vp_offs as u32;
                }
                if let Some(ref dr) = curve_render_data_ranges
                {
                    mapped.slice_mut(cd_offs * size_of::<BVertexLoopBlinnData>() + dr.data.start, m.b_vertex_loop_blinn_data.len())
                        .clone_from_slice(&m.b_vertex_loop_blinn_data);
                    mapped.slice_mut(cp_offs * size_of::<f32>() * 2 + dr.pos.start, m.b_vertex_positions.len())
                        .clone_from_slice(&m.b_vertex_positions);
                    cd_offs += m.b_vertex_loop_blinn_data.len();
                    cp_offs += m.b_vertex_positions.len();
                }
                vp_offs += m.b_quad_vertex_positions.len();
                ii_offs += m.b_quad_vertex_interior_indices.len();
            }
            if let Some(ref dr) = curve_render_data_ranges
            {
                mapped.slice_mut(dr.indices.start, bvlb_indices.len()).clone_from_slice(&bvlb_indices);
            }
            *mapped.get_mut(hint_const_range.start) = hint;
            mapped.slice_mut(transforms_range.start, transforms.len()).clone_from_slice(&transforms);
        }
        unsafe { umem.unmap(); }
        let init_commands = f.cmdpool.alloc(1, true).unwrap();
        init_commands[0].begin().unwrap()
            .pipeline_barrier(fe::PipelineStageFlags::TOP_OF_PIPE, fe::PipelineStageFlags::TRANSFER, true,
                &[], &[
                    fe::BufferMemoryBarrier::new(&buf, 0 .. prealloc.total, 0, fe::AccessFlags::TRANSFER.write),
                    fe::BufferMemoryBarrier::new(&ubuf, 0 .. prealloc.total, 0, fe::AccessFlags::TRANSFER.read)
                ], &[])
            .copy_buffer(&ubuf, &buf, &[fe::vk::VkBufferCopy { srcOffset: 0, dstOffset: 0, size: prealloc.total as _ }])
            .pipeline_barrier(fe::PipelineStageFlags::TRANSFER, fe::PipelineStageFlags::VERTEX_INPUT, true,
                &[], &[
                    fe::BufferMemoryBarrier::new(&buf, 0 .. prealloc.total, fe::AccessFlags::TRANSFER.write,
                        fe::AccessFlags::VERTEX_ATTRIBUTE_READ | fe::AccessFlags::INDEX_READ)
                ], &[]);
        f.queue.submit(&[fe::SubmissionBatch
        {
            command_buffers: Cow::Borrowed(&init_commands), .. Default::default()
        }], None)?;
        
        let descpool = fe::DescriptorPool::new(&f.device, 1, &[
            fe::DescriptorPoolSize(fe::DescriptorType::UniformTexelBuffer, 1),
            fe::DescriptorPoolSize(fe::DescriptorType::UniformBuffer, 1)
        ], false)?;
        let descset = descpool.alloc(&[utb_desc_layout])?.remove(0);
        let transforms_view =
            buf.create_view(fe::vk::VK_FORMAT_R32G32B32A32_SFLOAT, transforms_range.start as _ .. transforms_range.end as _)?;
        f.device.update_descriptor_sets(&[
            fe::DescriptorSetWriteInfo(descset, 0, 0,
                fe::DescriptorUpdateInfo::UniformTexelBuffer(vec![transforms_view.native_ptr()])),
            fe::DescriptorSetWriteInfo(descset, 1, 0,
                fe::DescriptorUpdateInfo::UniformBuffer(vec![(buf.native_ptr(), hint_const_range)]))
        ], &[]);

        f.device.wait()?;
        return Ok(PathfinderRenderBuffers
        {
            buf, mem, interior_pos_range, interior_indices_range, interior_vertex_range_per_mesh,
            bvlb_vertex_range_per_mesh, curve_render_data_ranges, descpool, descset, transforms_view
        })
    }
    fn populate_direct_render_commands_for(&self, rec: &mut fe::CmdRecord, size: &fe::Extent2D, res: &Resources)
    {
        use std::mem::size_of;

        rec .bind_graphics_pipeline_pair(&res.gp_simple_fill, &res.pl)
            .push_graphics_constant(fe::ShaderStage::VERTEX, 0, &[size.0 as f32, size.1 as _])
            .bind_graphics_descriptor_sets(0, &[self.descset], &[]);
        
        // draw interior
        rec.bind_vertex_buffers(0, &[(&self.buf, self.interior_pos_range.start)]);
        for (i, vcount) in self.interior_vertex_range_per_mesh.iter().enumerate().map(|(a, b)| (a as i32, b))
        {
            rec.bind_index_buffer(&self.buf, self.interior_indices_range.start + vcount.start * size_of::<u32>(), fe::IndexType::U32);
            rec.push_graphics_constant(fe::ShaderStage::VERTEX, PF_DIRECT_RENDER_GLYPH_INDEX_OFFSET, &i);
            rec.draw_indexed(vcount.len() as _, 1, 0, 0, 0);
        }
        // draw exterior curve(if provided)
        if let Some(ref dr) = self.curve_render_data_ranges
        {
            rec.bind_graphics_pipeline(&res.gp_curve_fill)
                .bind_vertex_buffers(0, &[(&self.buf, dr.pos.start), (&self.buf, dr.data.start)]);
            for (i, vcount) in self.bvlb_vertex_range_per_mesh.iter().enumerate().map(|(a, b)| (a as i32, b))
            {
                rec.bind_index_buffer(&self.buf, dr.indices.start + vcount.start * size_of::<u32>(), fe::IndexType::U32);
                rec.push_graphics_constant(fe::ShaderStage::VERTEX, PF_DIRECT_RENDER_GLYPH_INDEX_OFFSET, &i);
                rec.draw_indexed(vcount.len() as _, 1, 0, 0, 0);
            }
        }
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
        *self.ferrite.borrow_mut() = Some(Ferrite
        {
            fence_command_completion: fe::Fence::new(&device, false).unwrap(),
            semaphore_sync_next: fe::Semaphore::new(&device).unwrap(),
            semaphore_command_completion: fe::Semaphore::new(&device).unwrap(),
            tcmdpool: fe::CommandPool::new(&device, tq, false, false).unwrap(),
            cmdpool, queue, tqueue: device.queue(tq, if united_queue { 1 } else { 0 }),
            device, adapter, instance, gq, _tq: tq, _d: d, device_memindex, upload_memindex
        });

        let w = NativeWindowBuilder::new(640, 360, "Loop-Blinn Bezier Curve Drawing Example").create_renderable(server).unwrap();
        *self.w.borrow_mut() = Some(w);
        self.w.borrow().as_ref().unwrap().show();
    }
    fn on_init_view(&self, server: &GUIApplication<Self>, surface_onto: &NativeView<Self>)
    {
        let f = Ref::map(self.ferrite.borrow(), |o| o.as_ref().unwrap());

        if !server.presentation_support(&f.adapter, f.gq) { panic!("Vulkan Rendering is not supported by platform"); }
        let surface = server.create_surface(surface_onto, &f.instance).unwrap();
        if !f.adapter.surface_support(f.gq, &surface).unwrap() { panic!("Vulkan Rendering is not supported to this surface"); }
        let sd = SurfaceDesc::new(&surface, &f.adapter).unwrap();
        *self.res.borrow_mut() = Some(Resources::new(&self.ferrite_ref(), &sd).unwrap());
        let rtvs = WindowRenderTargets::new(&self.ferrite_ref(), &self.resources_ref(), &surface, &sd).unwrap().unwrap();
        *self.surface.borrow_mut() = Some((surface, sd));
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
                    self.ferrite_ref().fence_command_completion.wait().unwrap();
                    self.ferrite_ref().fence_command_completion.reset().unwrap();
                    *self.rcmds.borrow_mut() = None;
                    *self.rendertargets.borrow_mut() = None;

                    // reissue rendering
                    self.on_render_period();
                }
                else { let e: fe::Result<()> = Err(e); e.unwrap(); }
            }
        }
    }
}

macro_rules! DSLBindings {
    [$($binding: tt : [$kind: ident @ $stage: path; $count: expr]),*] => {
        fe::DSLBindings
        {
            $($kind: Some(($binding, $count, $stage)),)* ..fe::DSLBindings::empty()
        }
    }
}

struct Resources
{
    /*buf: fe::Buffer, _dmem: fe::DeviceMemory,*/
    gp_curve_fill: fe::Pipeline, gp_simple_fill: fe::Pipeline,
    pf_bufs: PathfinderRenderBuffers, pl: fe::PipelineLayout, rp: fe::RenderPass,
    #[allow(dead_code)] shaders: ShaderStore, #[allow(dead_code)] utb_desc_layout: fe::DescriptorSetLayout
}
impl Resources
{
    fn new(f: &Ferrite, sd: &SurfaceDesc) -> fe::Result<Self>
    {
        let mut rpb = fe::RenderPassBuilder::new();
        rpb.add_attachments(vec![
            fe::AttachmentDescription::new(sd.format.format, fe::ImageLayout::ColorAttachmentOpt, fe::ImageLayout::ColorAttachmentOpt)
                .load_op(fe::LoadOp::Clear).samples(SAMPLE_COUNT as _),
            fe::AttachmentDescription::new(sd.format.format, fe::ImageLayout::PresentSrc, fe::ImageLayout::PresentSrc)
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
        let rp = rpb.create(&f.device)?;

        let mut fc = FontContext::new().unwrap();
        /*let fontfile = std::fs::File::open("NotoSansCJKjp-Regular.otf")
            .and_then(|mut fp| { use std::io::prelude::*; let mut b = Vec::new(); fp.read_to_end(&mut b).map(|_| b) })
            .unwrap().into();
        fc.add_font_from_memory(&0, fontfile, 0).unwrap();*/
        // システムのデフォルトフォントを登録してみる
        let font = system_message_font_instance(&mut fc, 0);
        // FontInstanceのサイズ指定はデバイス依存px単位
        // let screen_dpi = screen_dpi();
        // println!("screen dpi: {}", screen_dpi);
        // let font = FontInstance::new(&0, Au::from_f32_px(13.0 * screen_dpi / 72.0));
        // text -> glyph indices
        let glyphs = fc.load_glyph_indices_for_characters(&font, &"Hello にゃーん".chars().map(|x| x as _).collect::<Vec<_>>()).unwrap();
        // glyph indices -> layouted text outlines
        let mut paths = Vec::new();
        let (mut left_offs, mut max_height) = (0.0, 0.0f32);
        for &g in &glyphs
        {
            let mut g = GlyphKey::new(g as _, SubpixelOffset(0));
            let dim = fc.glyph_dimensions(&font, &g, true).unwrap();
            // println!("dimension?: {:?}", dim);
            let rendered: f32 = left_offs;
            g.subpixel_offset.0 = (rendered.fract() * SUBPIXEL_GRANULARITY as f32) as _;
            if let Ok(outline) = fc.glyph_outline(&font, &g)
            {
                paths.push(Transform2DPathIter::new(outline.iter(),
                    &Transform2D::create_translation(rendered.trunc() as _, 0.0)).collect::<Vec<_>>());
            }
            left_offs += dim.advance/* * 60.0*/;
            max_height = max_height.max(dim.size.height as f32/* as i32 as f32 / 96.0*/);
        }
        /*{
            let mut g = GlyphKey::new(glyphs[1] as _, SubpixelOffset(0));
            let dim = fc.glyph_dimensions(&font, &g, false).unwrap();
            let rendered: f32 = left_offs;
            g.subpixel_offset.0 = (rendered.fract() * SUBPIXEL_GRANULARITY as f32) as _;
            let outline = fc.glyph_outline(&font, &g).unwrap();
            paths.extend(Transform2DPathIter::new(outline.iter(),
                &Transform2D::create_translation(rendered.trunc() as _, 0.0)));
            left_offs += dim.advance/* * 60.0*/;
            max_height = max_height.max(dim.size.height as f32/* as i32 as f32 / 96.0*/);
        }*/
        println!("left offset: {}", left_offs);
        println!("max height: {}", max_height);
        // text outlines -> pathfinder mesh
        let mut meshes = Vec::with_capacity(paths.len());
        for p in paths
        {
            let mut partitioner = Partitioner::new();
            for pe in Transform2DPathIter::new(p.into_iter(),
                &Transform2D::create_translation(-left_offs * 0.5, -max_height * 0.5))
            {
                partitioner.builder_mut().path_event(pe);
            }
            partitioner.partition(FillRule::Winding);
            partitioner.builder_mut().build_and_reset();
            meshes.push(partitioner.into_mesh());
        }
        // partitioner.mesh_mut().push_stencil_segments(CubicToQuadraticTransformer::new(paths.iter().cloned(), 5.0));
        // partitioner.mesh_mut().push_stencil_normals(CubicToQuadraticTransformer::new(paths.iter().cloned(), 5.0));

        let pixels_per_unit = fc.pixels_per_unit(&font).unwrap();
        let mut stem_darkening_offset = embolden_amount(font.size.to_f32_px(), pixels_per_unit);
        stem_darkening_offset[0] *= pixels_per_unit / 2.0f32.sqrt();
        stem_darkening_offset[1] *= pixels_per_unit / 2.0f32.sqrt();
        // println!("stem darkening offset: {:?}", stem_darkening_offset);
        let transforms = glyphs.iter().map(|_| GlyphTransform
        {
            st: [1.0, 1.0, stem_darkening_offset[0], stem_darkening_offset[1]], ext: [0.0; 2], pad: [0.0; 2]
        }).collect::<Vec<_>>();
        // println!("transform: {:?}", transforms);

        let utb_desc_layout = fe::DescriptorSetLayout::new(&f.device, &DSLBindings![
            0: [uniform_texel_buffer @ fe::ShaderStage::VERTEX; 1],
            1: [uniform_buffer @ fe::ShaderStage::VERTEX; 1]
        ])?;

        let shaders = ShaderStore::load(&f.device).unwrap();
        let pl = fe::PipelineLayout::new(&f.device, &[&utb_desc_layout], PF_DIRECT_RENDER_PUSH_CONSTANT_LAYOUT)?;
        let pf_bufs = PathfinderRenderBuffers::new(&f, &utb_desc_layout, meshes, transforms,
            Hint::compute(&mut fc, &font, true))?;

        let (gp_simple_fill, gp_curve_fill);
        {
            let mut ms = fe::MultisampleState::new();
            ms.rasterization_samples(SAMPLE_COUNT).sample_shading(Some(1.0)).sample_mask(&[(1 << SAMPLE_COUNT) - 1]);
            let mut gpb = fe::GraphicsPipelineBuilder::new(&pl, (&rp, 0));
            gpb .multisample_state(Some(&ms))
                .fixed_viewport_scissors(fe::DynamicArrayState::Dynamic(1), fe::DynamicArrayState::Dynamic(1))
                .add_attachment_blend(fe::AttachmentColorBlendState::premultiplied());
            
            let mut vps_simple = fe::VertexProcessingStages::new(fe::PipelineShader::new(&shaders.v_pass_fill, "main", None),
                VBIND_FILL_PF, VATTRS_FILL_PF, fe::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            vps_simple.fragment_shader(fe::PipelineShader::new(&shaders.f_white, "main", None));
            gpb.vertex_processing(vps_simple);
            gp_simple_fill = gpb.create(&f.device, None)?;

            let mut vps_curve = fe::VertexProcessingStages::new(fe::PipelineShader::new(&shaders.v_curve_pre, "main", None),
                VBIND_LB, VATTRS_LB, fe::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            vps_curve.fragment_shader(fe::PipelineShader::new(&shaders.f_white_curve, "main", None));
            gpb.vertex_processing(vps_curve);
            gp_curve_fill = gpb.create(&f.device, None)?;
        }

        return Ok(Resources { pf_bufs, gp_curve_fill, gp_simple_fill, pl, rp, shaders, utb_desc_layout })
    }
}
pub struct RenderCommands
{
    swapchain: Vec<fe::CommandBuffer>, #[allow(dead_code)] pathfinder_direct_render: Vec<fe::CommandBuffer>
}
impl App
{
    fn ferrite_ref(&self) -> Ref<Ferrite> { Ref::map(self.ferrite.borrow(), |f| f.as_ref().unwrap()) }
    fn resources_ref(&self) -> Ref<Resources> { Ref::map(self.res.borrow(), |r| r.as_ref().unwrap()) }

    fn ensure_render_targets(&self) -> fe::Result<bool>
    {
        if self.rendertargets.borrow().is_none()
        {
            let s2 = Ref::map(self.surface.borrow(), |r| r.as_ref().unwrap());
            let rtv = WindowRenderTargets::new(&self.ferrite_ref(), &self.resources_ref(), &s2.0, &s2.1)?;
            if rtv.is_none() { return Ok(false); }
            *self.rendertargets.borrow_mut() = rtv;
        }
        if self.rcmds.borrow().is_none()
        {
            *self.rcmds.borrow_mut() = Some(self.populate_render_commands().unwrap());
        }
        Ok(true)
    }
    fn populate_render_commands(&self) -> fe::Result<RenderCommands>
    {
        let (f, r) = (self.ferrite_ref(), self.resources_ref());
        let rtvs = Ref::map(self.rendertargets.borrow(), |v| v.as_ref().unwrap());

        let vp = fe::vk::VkViewport
        {
            x: 0.0, y: 0.0, width: rtvs.size.0 as _, height: rtvs.size.1 as _, minDepth: 0.0, maxDepth: 1.0
        };
        let scis = fe::vk::VkRect2D
        {
            offset: fe::vk::VkOffset2D { x: 0, y: 0 },
            extent: fe::vk::VkExtent2D { width: vp.width as _, height: vp.height as _ }
        };

        f.cmdpool.reset(true)?;
        let pf_drc = f.cmdpool.alloc(rtvs.framebuffers.len() as _, false)?;
        let render_commands = f.cmdpool.alloc(rtvs.framebuffers.len() as _, true)?;
        for (drc, (c, fb)) in pf_drc.iter().zip(render_commands.iter().zip(&rtvs.framebuffers))
        {
            {
                let mut drr = drc.begin_inherit(Some((fb, &r.rp, 0)), None)?;
                drr.set_viewport(0, &[vp.clone()]).set_scissor(0, &[scis.clone()]);
                r.pf_bufs.populate_direct_render_commands_for(&mut drr, fb.size(), &r);
            }
            let mut rec = c.begin()?;
            // rec.set_viewport(0, &[vp.clone()]).set_scissor(0, &[scis.clone()]);
            unsafe
            {
                rec.begin_render_pass(&r.rp, fb, fe::vk::VkRect2D
                {
                    offset: fe::vk::VkOffset2D { x: 0, y: 0 },
                    extent: fe::vk::VkExtent2D { width: rtvs.size.0, height: rtvs.size.1 }
                }, &[fe::ClearValue::Color([0.0, 0.0, 0.0, 1.0])], false)
                    .execute_commands(&[drc.native_ptr()]);
                rec.end_render_pass();
            }
        }
        Ok(RenderCommands { swapchain: render_commands, pathfinder_direct_render: pf_drc })
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
            command_buffers: Cow::Borrowed(&rcmds.swapchain[next..next+1]),
            wait_semaphores: Cow::Borrowed(&[(&f.semaphore_sync_next, fe::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)]),
            signal_semaphores: Cow::Borrowed(&[&f.semaphore_command_completion])
        }], Some(&f.fence_command_completion))?;
        f.queue.present(&[(&rtvs.swapchain, next as _)], &[&f.semaphore_command_completion])?;
        // コマンドバッファの使用が終了したことを明示する
        f.fence_command_completion.wait()?; f.fence_command_completion.reset()?; Ok(())
    }
}
struct SurfaceDesc
{
    format: fe::vk::VkSurfaceFormatKHR, present_mode: fe::PresentMode, composite_alpha: fe::CompositeAlpha
}
impl SurfaceDesc
{
    pub fn new(s: &fe::Surface, adapter: &fe::PhysicalDevice) -> fe::Result<Self>
    {
        let surface_caps = adapter.surface_capabilities(s)?;
        let format = adapter.surface_formats(s)?.into_iter()
            .find(|f| fe::FormatQuery(f.format).eq_bit_width(32).is_component_of(fe::FormatComponents::RGBA).has_element_of(fe::ElementType::UNORM).passed()).unwrap();
        let present_mode = adapter.surface_present_modes(s)?.remove(0);
        let composite_alpha = if (surface_caps.supportedCompositeAlpha & fe::CompositeAlpha::PostMultiplied as u32) != 0
        {
            fe::CompositeAlpha::PostMultiplied
        }
        else { fe::CompositeAlpha::Opaque };

        return Ok(SurfaceDesc { format, present_mode, composite_alpha });
    }
}
#[allow(dead_code)]
struct WindowRenderTargets
{
    framebuffers: Vec<fe::Framebuffer>, ms_dest: ImageMemoryPair, backbuffers: Vec<fe::ImageView>,
    swapchain: fe::Swapchain, size: fe::Extent2D
}
impl WindowRenderTargets
{
    fn new(f: &Ferrite, r: &Resources, s: &fe::Surface, sd: &SurfaceDesc) -> fe::Result<Option<Self>>
    {
        let surface_caps = f.adapter.surface_capabilities(s)?;
        let surface_size = match surface_caps.currentExtent
        {
            fe::vk::VkExtent2D { width: 0xffff_ffff, height: 0xffff_ffff } => fe::Extent2D(640, 360),
            fe::vk::VkExtent2D { width, height } => fe::Extent2D(width, height)
        };
        if surface_size.0 <= 0 || surface_size.1 <= 0 { return Ok(None); }
        let swapchain = fe::SwapchainBuilder::new(s, surface_caps.minImageCount.max(2),
            &sd.format, &surface_size, fe::ImageUsage::COLOR_ATTACHMENT)
                .present_mode(sd.present_mode).pre_transform(fe::SurfaceTransform::Identity)
                .composite_alpha(sd.composite_alpha).create(&f.device)?;
        // acquire_nextより前にやらないと死ぬ(get_images)
        let backbuffers = swapchain.get_images()?;
        let isr = fe::ImageSubresourceRange::color(0, 0);
        let bb_views = backbuffers.iter().map(|i| i.create_view(None, None, &fe::ComponentMapping::default(), &isr))
            .collect::<fe::Result<Vec<_>>>()?;
        
        let ms_dest = fe::ImageDesc::new(&surface_size, sd.format.format,
            fe::ImageUsage::COLOR_ATTACHMENT.transient_attachment(),
            fe::ImageLayout::Undefined).sample_counts(SAMPLE_COUNT as _).create(&f.device)?;
        let ms_dest = ImageMemoryPair::new(ms_dest, f.device_memindex)?;

        let framebuffers = bb_views.iter().map(|iv| fe::Framebuffer::new(&r.rp, &[&ms_dest.view, iv], &surface_size, 1))
            .collect::<fe::Result<Vec<_>>>()?;

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
        }], None)?; f.queue.wait()?;
        
        return Ok(Some(WindowRenderTargets
        {
            swapchain, backbuffers: bb_views, framebuffers, size: surface_size, ms_dest
        }));
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

fn compute_stem_darkening_amount(pixels_per_em: f32, pixels_per_unit: f32) -> [f32; 2]
{
    const LIMIT_SIZE: f32 = 72.0;
    let amounts: [f32; 2] = [(0.0121 * 2.0f32.sqrt()) * (2.0 / screen_multiplier()),
        (0.0121 * 1.25 * 2.0f32.sqrt()) * (2.0 / screen_multiplier())];

    if pixels_per_em <= LIMIT_SIZE
    {
        let scaled_amount = |a| f32::min(a * pixels_per_em, LIMIT_SIZE) / pixels_per_unit;
        [scaled_amount(amounts[0]), scaled_amount(amounts[1])]
    }
    else { [0.0; 2] }
}
#[cfg(feature = "StemDarkening")]
fn stem_darkening_amount(font_size: f32, pixels_per_unit: f32) -> [f32; 2]
{
    compute_stem_darkening_amount(font_size, pixels_per_unit)
}
#[cfg(not(feature = "StemDarkening"))]
fn stem_darkening_amount(_font_size: f32, _pixels_per_unit: f32) -> [f32; 2] { [0.0; 2] }
fn embolden_amount(font_size: f32, pixels_per_unit: f32) -> [f32; 2] { stem_darkening_amount(font_size, pixels_per_unit) }

#[cfg(target_os = "macos")] #[macro_use] extern crate objc;
#[cfg(target_os = "macos")] extern crate core_graphics;
#[cfg(target_os = "macos")] extern crate core_text;
#[cfg(target_os = "macos")] extern crate foreign_types_shared;
#[cfg(target_os = "macos")] use objc::runtime::*;
#[cfg(target_os = "macos")]
#[cfg(target_pointer_width = "64")] #[repr(C)] #[derive(Debug, Clone, Copy)] struct CGFloat(f64);
#[cfg(target_os = "macos")]
#[cfg(target_pointer_width = "32")] #[repr(C)] #[derive(Debug, Clone, Copy)] struct CGFloat(f32);
#[cfg(target_os = "macos")] fn screen_dpi() -> f32
{
    let screen: *mut Object = unsafe { msg_send![Class::get("NSScreen").unwrap(), mainScreen] };
    let backing_scale_factor: CGFloat = unsafe { msg_send![screen, backingScaleFactor] };
    return (72.0 * backing_scale_factor.0) as f32;
}
#[cfg(not(target_os = "macos"))] fn screen_dpi() -> f32 { 72.0 }

#[cfg(not(target_os = "macos"))] fn screen_multiplier() -> f32 { 1.0 }

#[cfg(target_os = "macos")] fn system_message_font_instance(fc: &mut FontContext<usize>, key: usize) -> FontInstance<usize>
{
    use core_graphics::font::CGFont;
    use foreign_types_shared::ForeignType;
    use libc::{c_void, /*c_long, c_char*/}; use std::ptr::null_mut;
    extern "system"
    {
        fn CTFontCopyGraphicsFont(font: *mut c_void, attributes: *mut c_void) -> *mut c_void;
        /*fn CTFontCopySupportedLanguages(font: *mut c_void) -> *mut c_void;
        fn CFArrayGetCount(array: *const c_void) -> c_long;
        fn CFArrayGetValueAtIndex(array: *const c_void, index: c_long) -> *const c_void;
        fn CFStringGetCStringPtr(string: *const c_void, encoding: u32) -> *const c_char;
        fn CFStringGetCString(string: *const c_void, buf: *mut c_char, bufferSize: c_long, encoding: u32) -> bool; */
    }
    // const kCFStringEncodingUTF8: u32 = 0x0800_0100;

    // システムフォント(San Francisco/Helvetica Neue)は日本語に対応していない
    // そのうちフォールバック機能をfont-rendererにつける必要がある
    let fontname: *mut Object = unsafe { msg_send![Class::get("NSString").unwrap(), stringWithUTF8String: "ヒラギノ角ゴシック W3\0".as_ptr()] };
    let nsfont: *mut Object = unsafe { msg_send![Class::get("NSFont").unwrap(), fontWithName: fontname size: CGFloat(0.0)] };
    let _: () = unsafe { msg_send![fontname, release] };
    let point_size: CGFloat = unsafe { msg_send![nsfont, pointSize] };
    let cgfont = unsafe { CGFont::from_ptr(CTFontCopyGraphicsFont(nsfont as *mut _, null_mut()) as *mut _) };
    /*let languages = unsafe { CTFontCopySupportedLanguages(nsfont as *mut _) };
    for n in 0 .. unsafe { CFArrayGetCount(languages) }
    {
        let sref = unsafe { CFArrayGetValueAtIndex(languages, n) };
        let mut cstr = [0; 256];
        unsafe { CFStringGetCString(sref, cstr.as_mut_ptr(), 256, kCFStringEncodingUTF8); }
        println!("* {}", unsafe { std::ffi::CStr::from_ptr(cstr.as_ptr()).to_str().unwrap() });
    }*/
    fc.add_native_font(&key, cgfont).unwrap();
    return FontInstance::new(&key, Au::from_f32_px(point_size.0 as f32 * screen_dpi() / 72.0));
}
#[cfg(windows)] extern crate winapi;
#[cfg(windows)] fn system_message_font_instance(fc: &mut FontContext<usize>, key: usize) -> FontInstance<usize>
{
    use winapi::um::winuser::*;

    // Windowsのシステムフォントはおおよそ日本語に対応してくれているのでそのまま使える
    let mut ncm = NONCLIENTMETRICSA
    {
        cbSize: std::mem::size_of::<NONCLIENTMETRICSA>() as _,
        .. unsafe { std::mem::uninitialized() }
    };
    unsafe { SystemParametersInfoA(SPI_GETNONCLIENTMETRICS, ncm.cbSize, &mut ncm as *mut NONCLIENTMETRICSA as *mut _, 0) };
    println!("font size: {}", -ncm.lfMessageFont.lfHeight);
    let font_name = unsafe { std::ffi::CStr::from_ptr(ncm.lfMessageFont.lfFaceName.as_ptr()).to_str().unwrap() };
    fc.add_system_font(&key, font_name, 0).unwrap();
    return FontInstance::new(&key, Au::from_px(-ncm.lfMessageFont.lfHeight));
}
