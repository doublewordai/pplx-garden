use std::{
    ffi::c_void,
    ptr::{null, null_mut},
};

use p2p_all_to_all::{AllToAllContext, AllToAllRankHandle, LowLatencyRouteLayoutPlan};
use pyo3::{
    Bound, PyResult, Python, exceptions::PyRuntimeError, pyclass, pymethods,
    types::PyDict, types::PyDictMethods, types::PyModule, types::PyModuleMethods,
};
use torch_lib::ScalarType;

use crate::py_fabric_lib::{
    PyDomainAddress, PyMemoryRegionDescriptor, PyMemoryRegionHandle, PyTransferEngine,
};

fn low_latency_route_layout_plan_to_dict<'py>(
    py: Python<'py>,
    plan: LowLatencyRouteLayoutPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("source_group_order", plan.source_group_order)?;
    dict.set_item("source_rank", plan.source_rank)?;
    dict.set_item("source_group", plan.source_group)?;
    dict.set_item("final_index", plan.final_index)?;
    dict.set_item(
        "tokens_per_source_group_per_local_expert",
        plan.tokens_per_source_group_per_local_expert,
    )?;
    dict.set_item("tokens_per_expert", plan.tokens_per_expert)?;
    dict.set_item("num_recv_tokens", plan.num_recv_tokens)?;
    Ok(dict)
}

#[pyclass(name = "AllToAllContext", module = "pplx_garden._rust")]
pub(crate) struct PyAllToAllContext {
    ctx: AllToAllContext,
}

#[pymethods]
impl PyAllToAllContext {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn create(
        hidden_dim: usize,
        hidden_dim_scale: Option<usize>,
        in_elemsize: usize,
        out_elemsize: usize,
        out_dtype: ScalarType,
        scale_elemsize: Option<usize>,
        max_num_tokens: usize,
        max_recv_tokens: usize,
        max_tokens_per_expert: Option<usize>,
        max_private_tokens: usize,
        num_experts: usize,
        expert_padding: usize,
        num_experts_per_token: usize,
        rank: usize,
        dp_size: usize,
        node_size: usize,
        world_size: usize,
        num_routed_ptrs: Vec<u64>,
        num_routed_mrs: Vec<PyMemoryRegionHandle>,
        send_buffer_ptrs: Vec<u64>,
        send_buffer_mrs: Vec<PyMemoryRegionHandle>,
        recv_buffer_ptrs: Vec<u64>,
        recv_buffer_mrs: Vec<PyMemoryRegionHandle>,
        sync_ptrs: Vec<Vec<u64>>,
        send_ptrs: Vec<Vec<u64>>,
        recv_ptrs: Vec<Vec<u64>>,
        node_route_count_ptrs: Vec<Vec<u64>>,
        node_route_epoch_ptrs: Vec<Vec<u64>>,
        device: u8,
        imm_base: u32,
        ranks: Vec<(
            PyDomainAddress,
            Vec<PyMemoryRegionDescriptor>,
            Vec<PyMemoryRegionDescriptor>,
        )>,
        transfer_engine: &PyTransferEngine,
        worker_cpu: Option<u16>,
        num_slots: usize,
    ) -> PyResult<Self> {
        let mut rank_handles = vec![Vec::with_capacity(ranks.len()); num_slots.max(1)];
        for (address, num_routed_descs, recv_buffer_descs) in ranks {
            if num_routed_descs.len() != rank_handles.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "Expected {} num_routed descriptors, got {}",
                    rank_handles.len(),
                    num_routed_descs.len()
                )));
            }
            if recv_buffer_descs.len() != rank_handles.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "Expected {} recv buffer descriptors, got {}",
                    rank_handles.len(),
                    recv_buffer_descs.len()
                )));
            }
            for (slot, (num_routed_desc, recv_buffer_desc)) in num_routed_descs
                .into_iter()
                .zip(recv_buffer_descs.into_iter())
                .enumerate()
            {
                rank_handles[slot].push(AllToAllRankHandle::new(
                    address.0.clone(),
                    num_routed_desc.0,
                    recv_buffer_desc.0,
                ));
            }
        }

        let ctx = AllToAllContext::new(
            hidden_dim,
            hidden_dim_scale.unwrap_or(0),
            in_elemsize,
            out_elemsize,
            out_dtype,
            scale_elemsize.unwrap_or(0),
            max_num_tokens,
            max_recv_tokens,
            max_tokens_per_expert.unwrap_or(0),
            max_private_tokens,
            num_experts,
            expert_padding,
            num_experts_per_token,
            rank,
            dp_size,
            node_size,
            world_size,
            num_routed_ptrs.into_iter().map(|ptr| ptr as *mut u32).collect(),
            num_routed_mrs.into_iter().map(|mr| mr.0).collect(),
            send_buffer_ptrs.into_iter().map(|ptr| ptr as *mut c_void).collect(),
            send_buffer_mrs.into_iter().map(|mr| mr.0).collect(),
            recv_buffer_ptrs.into_iter().map(|ptr| ptr as *mut c_void).collect(),
            recv_buffer_mrs.into_iter().map(|mr| mr.0).collect(),
            sync_ptrs,
            send_ptrs,
            recv_ptrs,
            node_route_count_ptrs,
            node_route_epoch_ptrs,
            device,
            imm_base,
            rank_handles,
            transfer_engine.get_fabric_engine(),
            worker_cpu,
            num_slots,
        )?;
        Ok(Self { ctx })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_send<'py>(
        &mut self,
        py: Python<'py>,
        num_tokens: usize,
        x_ptr: u64,
        x_stride: usize,
        x_scale_ptr: Option<u64>,
        x_scale_stride_elem: Option<usize>,
        x_scale_stride_token: Option<usize>,
        indices_ptr: u64,
        indices_stride: usize,
        weights_ptr: u64,
        weights_stride: usize,
        bound_m_ptr: Option<u64>,
        stream: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let handle = self
            .ctx
            .dispatch_send(
                num_tokens,
                x_ptr as *const c_void,
                x_stride,
                x_scale_ptr.map(|ptr| ptr as *const c_void).unwrap_or(null()),
                x_scale_stride_elem.unwrap_or(0),
                x_scale_stride_token.unwrap_or(0),
                indices_ptr as *const i32,
                indices_stride,
                weights_ptr as *const f32,
                weights_stride,
                bound_m_ptr.map(|ptr| ptr as *const i32).unwrap_or(null()),
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("slot", handle.slot)?;
        dict.set_item("generation", handle.generation)?;
        Ok(dict)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_send_on_slot<'py>(
        &mut self,
        py: Python<'py>,
        slot: usize,
        num_tokens: usize,
        x_ptr: u64,
        x_stride: usize,
        x_scale_ptr: Option<u64>,
        x_scale_stride_elem: Option<usize>,
        x_scale_stride_token: Option<usize>,
        indices_ptr: u64,
        indices_stride: usize,
        weights_ptr: u64,
        weights_stride: usize,
        bound_m_ptr: Option<u64>,
        stream: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let handle = self
            .ctx
            .dispatch_send_on_slot(
                slot,
                num_tokens,
                x_ptr as *const c_void,
                x_stride,
                x_scale_ptr.map(|ptr| ptr as *const c_void).unwrap_or(null()),
                x_scale_stride_elem.unwrap_or(0),
                x_scale_stride_token.unwrap_or(0),
                indices_ptr as *const i32,
                indices_stride,
                weights_ptr as *const f32,
                weights_stride,
                bound_m_ptr.map(|ptr| ptr as *const i32).unwrap_or(null()),
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("slot", handle.slot)?;
        dict.set_item("generation", handle.generation)?;
        Ok(dict)
    }

    fn low_latency_workspace_layout<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let layout = self.ctx.low_latency_workspace_layout();
        let dict = PyDict::new(py);
        let tensors = PyDict::new(py);
        dict.set_item("alignment", layout.alignment)?;
        dict.set_item("total_bytes", layout.total_bytes)?;
        for tensor in layout.tensors {
            let spec = PyDict::new(py);
            spec.set_item("offset_bytes", tensor.offset_bytes)?;
            spec.set_item("nbytes", tensor.nbytes)?;
            spec.set_item("shape", tensor.shape)?;
            spec.set_item("dtype", tensor.dtype)?;
            tensors.set_item(tensor.name, spec)?;
        }
        dict.set_item("tensors", tensors)?;
        Ok(dict)
    }

    fn debug_low_latency_route_layout_plan<'py>(
        &self,
        py: Python<'py>,
        num_routed: Vec<Vec<u32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let plan = self
            .ctx
            .debug_low_latency_route_layout_plan(num_routed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        low_latency_route_layout_plan_to_dict(py, plan)
    }

    fn debug_low_latency_route_layout_plan_for_handle<'py>(
        &self,
        py: Python<'py>,
        slot: usize,
        generation: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let plan = self
            .ctx
            .debug_low_latency_route_layout_plan_for_handle(slot, generation)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        low_latency_route_layout_plan_to_dict(py, plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_recv(
        &mut self,
        slot: usize,
        generation: u64,
        out_num_tokens_ptr: u64,
        out_x_ptr: u64,
        out_x_stride: usize,
        out_x_scale_ptr: Option<u64>,
        out_x_scale_stride_elem: Option<usize>,
        out_x_scale_stride_token: Option<usize>,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .dispatch_recv(
                slot,
                generation,
                out_num_tokens_ptr as *mut i32,
                out_x_ptr as *mut c_void,
                out_x_stride,
                out_x_scale_ptr.map(|ptr| ptr as *mut c_void).unwrap_or(null_mut()),
                out_x_scale_stride_elem.unwrap_or(0),
                out_x_scale_stride_token.unwrap_or(0),
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_send(
        &mut self,
        slot: usize,
        generation: u64,
        expert_x_ptr: u64,
        expert_x_stride: usize,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .combine_send(
                slot,
                generation,
                expert_x_ptr as *const c_void,
                expert_x_stride,
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_recv(
        &mut self,
        slot: usize,
        generation: u64,
        num_tokens: usize,
        num_recv_tokens: usize,
        expert_y_dtype: ScalarType,
        out_tokens_ptr: u64,
        out_tokens_stride: usize,
        indices_ptr: u64,
        indices_stride: usize,
        weights_ptr: u64,
        weights_stride: usize,
        bound_m_ptr: Option<u64>,
        accumulate: bool,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .combine_recv(
                slot,
                generation,
                num_tokens,
                num_recv_tokens,
                expert_y_dtype,
                out_tokens_ptr as *mut c_void,
                out_tokens_stride,
                indices_ptr as *const i32,
                indices_stride,
                weights_ptr as *const f32,
                weights_stride,
                bound_m_ptr.map(|ptr| ptr as *const i32).unwrap_or(null()),
                accumulate,
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get_perf_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.ctx.get_perf_stats();
        let dict = PyDict::new(py);
        dict.set_item("local_dispatch_bytes", stats.local_dispatch_bytes)?;
        dict.set_item("nvlink_dispatch_bytes", stats.nvlink_dispatch_bytes)?;
        dict.set_item("network_dispatch_bytes", stats.network_dispatch_bytes)?;
        dict.set_item("local_combine_bytes", stats.local_combine_bytes)?;
        dict.set_item("nvlink_combine_bytes", stats.nvlink_combine_bytes)?;
        dict.set_item("network_combine_bytes", stats.network_combine_bytes)?;
        dict.set_item("peer_dispatch_bytes", &stats.peer_dispatch_bytes)?;
        dict.set_item("peer_combine_bytes", &stats.peer_combine_bytes)?;
        dict.set_item("wait_dispatch_route_ns", stats.wait_dispatch_route_ns)?;
        dict.set_item("route_exchange_ns", stats.route_exchange_ns)?;
        dict.set_item("process_routing_ns", stats.process_routing_ns)?;
        dict.set_item("wait_dispatch_send_ns", stats.wait_dispatch_send_ns)?;
        dict.set_item("dispatch_transfer_wait_ns", stats.dispatch_transfer_wait_ns)?;
        dict.set_item("wait_dispatch_recv_ns", stats.wait_dispatch_recv_ns)?;
        dict.set_item("dispatch_barrier_ns", stats.dispatch_barrier_ns)?;
        dict.set_item("wait_combine_send_ns", stats.wait_combine_send_ns)?;
        dict.set_item("combine_transfer_wait_ns", stats.combine_transfer_wait_ns)?;
        dict.set_item("wait_combine_recv_ns", stats.wait_combine_recv_ns)?;
        dict.set_item("combine_barrier_ns", stats.combine_barrier_ns)?;
        Ok(dict)
    }
}

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAllToAllContext>()?;
    Ok(())
}
