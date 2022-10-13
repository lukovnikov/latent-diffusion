"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import noise_like


def make_ddim_timesteps(ddim_discr_method, num_ddim_jumps, total_steps, verbose=True, version="addfinal"):
    if version == "addfinal":
        num_ddim_jumps -= 1

    if ddim_discr_method == 'uniform':
        # c = total_steps // num_ddim_jumps
        # ddim_timesteps = list(range(0, total_steps, c))
        ddim_timesteps = list(range(0, num_ddim_jumps))
        ddim_timesteps = [round(xi * total_steps/num_ddim_jumps) for xi in ddim_timesteps]
    elif ddim_discr_method == 'quad':
        ddim_timesteps = list(((np.linspace(0, np.sqrt(total_steps * .8), num_ddim_jumps)) ** 2).astype(int))
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
    ddim_timesteps = ddim_timesteps + [total_steps]

    # pair up
    timesteps = list(zip(ddim_timesteps[1:], ddim_timesteps[:-1]))
    timesteps = timesteps[::-1]

    if version == "openai":
        # OpenAI version
        timesteps = timesteps[1:-1]
        timesteps += [(timesteps[-1][1], 1), (1, 0)]

    elif version == "addfinal":     # add one more timestep at the end
        timesteps = timesteps[:-1]
        timesteps += [(timesteps[-1][1], 1), (1, 0)]

    return timesteps


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.cachehash = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        cachehash = f"{ddim_num_steps}{ddim_eta}"  # TODO add model hash here
        if cachehash == self.cachehash:
            return

        self.cachehash = cachehash
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_', torch.cat([self.alphas_cumprod_prev[0:1], self.alphas_cumprod], 0))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        # self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
        #                                           num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        # ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
        #                                                                            ddim_timesteps=self.ddim_timesteps,
        #                                                                            eta=ddim_eta,verbose=verbose)
        # self.register_buffer('ddim_sigmas', ddim_sigmas)
        # self.register_buffer('ddim_alphas', ddim_alphas)
        # self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        # self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ddimmode="normal",
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        timesteps = make_ddim_timesteps("uniform", S, self.ddpm_num_timesteps, version=ddimmode)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    timesteps=timesteps,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            xt = torch.randn(shape, device=device)
        else:
            xt = x_T

        xt_acc = []
        pred_x0_acc = []

        print(f"Running DDIM Sampling with {len(timesteps)} timesteps")

        for i, (jump_start_t, jump_end_t) in enumerate(tqdm(timesteps, desc='DDIM Sampler', total=len(timesteps))):
            ts = torch.full((b,), jump_start_t-1, device=device, dtype=torch.long)      # jump_start_t - 1 because jump_start_t is in regular time range (ends at 1000)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                xt = img_orig * mask + (1. - mask) * xt

            outs = self.p_sample_ddim(xt, cond, ts, jump_start_t, jump_end_t,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            xt, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            xt_acc.append(xt)
            pred_x0_acc.append(pred_x0)

        return xt, {"xts": xt_acc, "pred_x0s": pred_x0_acc}

    @torch.no_grad()
    def p_sample_ddim(self, xt, cond, time_inp, jump_start_t, jump_end_t, repeat_noise=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *xt.shape, xt.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(xt, time_inp, cond)
        else:
            x_in = torch.cat([xt] * 2)
            t_in = torch.cat([time_inp] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, xt, time_inp, cond, **corrector_kwargs)

        # select parameters corresponding to the currently considered timestep
        # NOTE:
        # self.alphas_cumprod_ (note: underscore) corresponds to normal (as in paper) indexes, so T=1000, t=0 is end result)
        # jump_start_t and jump_end_t are also using these normal indexes
        a_t = torch.full((b, 1, 1, 1), self.alphas_cumprod_[jump_start_t], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.alphas_cumprod_[jump_end_t], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas_for_original_num_steps[jump_start_t-1], device=device)

        # current prediction for x_0
        pred_x0 = (xt - (1 - a_t).sqrt() * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
