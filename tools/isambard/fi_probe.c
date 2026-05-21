#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump(const char *label, struct fi_info *hints) {
  struct fi_info *info = NULL;
  int ret = fi_getinfo(FI_VERSION(1, 22), NULL, NULL, 0, hints, &info);
  printf("%s ret=%d", label, ret);
  if (ret != 0) {
    printf(" %s\n", fi_strerror(-ret));
    return;
  }
  int n = 0;
  for (struct fi_info *p = info; p != NULL; p = p->next) {
    struct fi_info *q = fi_dupinfo(p);
    struct fid_fabric *fabric = NULL;
    struct fid_domain *domain = NULL;
    struct fid_ep *ep = NULL;
    int fabric_ret = fi_fabric(q->fabric_attr, &fabric, NULL);
    int domain_ret = fabric_ret ? fabric_ret : fi_domain(fabric, q, &domain, NULL);
    int ep_ret = domain_ret ? domain_ret : fi_endpoint(domain, q, &ep, NULL);
    printf("\n  provider=%s fabric=%s domain=%s caps=0x%lx mode=0x%lx ep=%d mr=0x%lx threading=%d",
           p->fabric_attr && p->fabric_attr->prov_name ? p->fabric_attr->prov_name : "?",
           p->fabric_attr && p->fabric_attr->name ? p->fabric_attr->name : "?",
           p->domain_attr && p->domain_attr->name ? p->domain_attr->name : "?",
           p->caps,
           p->mode,
           p->ep_attr ? p->ep_attr->type : -1,
           p->domain_attr ? p->domain_attr->mr_mode : 0UL,
           p->domain_attr ? p->domain_attr->threading : -1);
    printf(" fabric_ret=%d domain_ret=%d endpoint_ret=%d", fabric_ret, domain_ret, ep_ret);
    if (ep) fi_close(&ep->fid);
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    fi_freeinfo(q);
    n++;
  }
  printf("\n  count=%d\n", n);
  fi_freeinfo(info);
}

int main(void) {
  printf("FI_PROVIDER=%s\n", getenv("FI_PROVIDER"));
  int ndev = -1;
  cudaError_t cerr = cudaGetDeviceCount(&ndev);
  printf("cudaGetDeviceCount=%d ndev=%d\n", (int)cerr, ndev);
  dump("null", NULL);

  struct fi_info *h = fi_allocinfo();
  h->ep_attr->type = FI_EP_RDM;
  dump("ep_rdm", h);

  h->caps = FI_MSG | FI_RMA;
  dump("ep_rdm_msg_rma", h);

  h->fabric_attr->prov_name = strdup("cxi");
  dump("ep_rdm_msg_rma_prov", h);

  free(h->fabric_attr->prov_name);
  h->fabric_attr->prov_name = NULL;
  fi_freeinfo(h);
  return 0;
}
