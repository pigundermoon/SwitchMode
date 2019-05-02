/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPicYuv.cpp
	\brief    picture YUV buffer class
	*/

#include <cstdlib>
#include <assert.h>
#include <memory.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include "TComPicYuv.h"
#include "TLibVideoIO/TVideoIOYuv.h"

//! \ingroup TLibCommon
//! \{

TComPicYuv::TComPicYuv()
{
	for (UInt i = 0; i < MAX_NUM_COMPONENT; i++)
	{
		m_apiPicBuf[i] = NULL;   // Buffer (including margin)
		m_piPicOrg[i] = NULL;    // m_apiPicBufY + m_iMarginLuma*getStride() + m_iMarginLuma
	}

#if enable_deep_interpolation
	m_PicYHalfIntered = NULL;
	ifedited = false;
#if place_RDO
	m_otherside_PicYIntered = NULL;
	ifplaceedited = false;
#endif
#endif

	for (UInt i = 0; i < MAX_NUM_CHANNEL_TYPE; i++)
	{
		m_ctuOffsetInBuffer[i] = 0;
		m_subCuOffsetInBuffer[i] = 0;
	}

	m_bIsBorderExtended = false;
}




TComPicYuv::~TComPicYuv()
{
}

#if enable_deep_interpolation
Pel* TComPicYuv::getInterAddr(const Int ctuRSAddr, const Int uiAbsZorderIdx, int qp, bool ifcheck)
{
	const ComponentID ch = ComponentID(0);

	if (m_PicYHalfIntered == NULL || ifedited)
	{
		if (m_PicYHalfIntered == NULL)
		{
			m_PicYHalfIntered = (Pel*)xMalloc(Pel, getStride(ch) * 4 * getTotalHeight(ch) * 4);
		}

		string basetmodelstr = "testH22.caffemodel";

		int tbestqp = 22;
		int qps[4] = { 22, 27, 32, 37 };
		for (int i = 0; i < 4; i++)
		{
			if (abs(qp - tbestqp)>abs(qp - qps[i]))
			{
				tbestqp = qps[i];
			}
		}
		//cout << qp << tbestqp << endl;
		basetmodelstr[5] = char(tbestqp / 10 + 48);
		basetmodelstr[6] = char(tbestqp % 10 + 48);

		Pel* srcptr = m_apiPicBuf[ch];
		Pel* dstptr = m_PicYHalfIntered;
		cv::Mat_<float> tmpimg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_32FC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg(i, j) = srcptr[j] / 255.0;
			}
			srcptr += getStride(ch);
		}

		//string model_file1 = "check_test_4x.prototxt";
		//string trained_file1 = "check_test_4x.caffemodel";
		//Interpolator interpolatort(model_file1, trained_file1, false);
		//interpolatort.interpolate_test();


		cv::Mat_<Short> resimg = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch), CV_16SC1);
		cv::Mat_<Short> tmpimg1 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
		std::vector<cv::Mat>  predictions;
		string model_file = "fast_interp_deploy_2x_l20.prototxt";

		clock_t start_time = clock();
		Interpolator interpolator(model_file, basetmodelstr, false);
		predictions = interpolator.interpolate(tmpimg);
		clock_t end_time = clock();
		printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

		tmpimg1 = predictions[0] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				resimg(i * 4, j * 4) = tmpimg(i, j) * 255 * 64;
				resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[1] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[2] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			}
		}

		inter_type inter_type_flag;

		inter_type_flag = sep_inone_small_12;//sep_inone_small_4x_c_ref sep_inone_small sep_inone_small_4x_c sep_inone_small direct_inter_fat direct_inter sep_inter_yuv sep_inone_small sep_inone_fat

		if (inter_type_flag == sep_inone_small_4x_c_ref)
		{
			model_file = "mfi_x4_c_ref_net.prototxt";
			basetmodelstr[4] = 'Q';

			start_time = clock();
			Interpolator interpolator_4x(model_file, basetmodelstr, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_num(tmpimg, 14);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 - 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 - 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 - 1) = tmpimg1(i, j);
				}
			}

			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}

			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			//tmpimg1 = predictions[8] * 255 * 64;
			//for (int i = 1; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 - 1, j * 4) = tmpimg1(i, j);
			//	}
			//}

			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 - 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 - 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 2, j * 4 - 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inone_small_12)
		{
			model_file = "mfi_x4_initial_12_deploy.prototxt";
			basetmodelstr[4] = 'Q';

			start_time = clock();
			Interpolator interpolator_4x(model_file, basetmodelstr, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x_c(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[1] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			//tmpimg1 = predictions[7] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[9] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inone_small)
		{
			model_file = "mfi_x4_initial_deploy.prototxt";
			basetmodelstr[4] = 'Q';

			start_time = clock();
			Interpolator interpolator_4x(model_file, basetmodelstr, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//0,2
			//tmpimg1 = predictions[1] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			//2,0
			//tmpimg1 = predictions[7] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			//2,2
			//tmpimg1 = predictions[9] * 255 * 64;
			//for (int i = 0; i < getTotalHeight(ch); i++)
			//{
			//	for (int j = 0; j < getStride(ch); j++)
			//	{
			//		resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			//	}
			//}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[12] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[13] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[14] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 3, j * 4 + 3) = tmpimg1(i, j);
				}
			}
		}
		else if (inter_type_flag == sep_inone_small_4x_c)
		{
			model_file = "mfi_x4_c_15_net.prototxt";
			basetmodelstr[4] = 'Q';

			start_time = clock();
			Interpolator interpolator_4x(model_file, basetmodelstr, false);
			predictions.clear();
			predictions = interpolator_4x.interpolate_4x(tmpimg);
			end_time = clock();
			printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

			tmpimg1 = predictions[0] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[1] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4, j * 4 - 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[2] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[3] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[4] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 - 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[5] * 255 * 64;
			for (int i = 0; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 + 1, j * 4 - 1) = tmpimg1(i, j);
				}
			}

			tmpimg1 = predictions[6] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 2, j * 4 + 1) = tmpimg1(i, j);
				}
			}

			tmpimg1 = predictions[7] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 2, j * 4 - 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[8] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[9] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 0; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 + 1) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[10] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 - 2) = tmpimg1(i, j);
				}
			}
			tmpimg1 = predictions[11] * 255 * 64;
			for (int i = 1; i < getTotalHeight(ch); i++)
			{
				for (int j = 1; j < getStride(ch); j++)
				{
					resimg(i * 4 - 1, j * 4 - 1) = tmpimg1(i, j);
				}
			}
		}

		for (int i = 0; i < resimg.rows; i++)
		{
			for (int j = 0; j < resimg.cols; j++)
			{
				if (resimg(i, j) >= 255 * 64)
				{
					resimg(i, j) = 255 * 64;
				}
				if (resimg(i, j) < 0)
				{
					resimg(i, j) = 0;
				}
			}
		}
		cv::Mat tmp;
		resimg.convertTo(tmp, CV_16UC1);
		tmp = tmp / 64;

		cv::Mat tmpchar;
		tmp.convertTo(tmpchar, CV_8UC1);

		cv::Mat_<uchar> tmpchar_2x = cv::Mat(2 * getTotalHeight(ch), 2 * getStride(ch), CV_16SC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpchar_2x(i * 2, j * 2) = resimg(i * 4, j * 4) / 64;
				tmpchar_2x(i * 2 + 1, j * 2) = resimg(i * 4 + 2, j * 4) / 64;
				tmpchar_2x(i * 2, j * 2 + 1) = resimg(i * 4, j * 4 + 2) / 64;
				tmpchar_2x(i * 2 + 1, j * 2 + 1) = resimg(i * 4 + 2, j * 4 + 2) / 64;
			}
		}


		predictions.clear();
		tmpimg.release();
		tmpimg1.release();


		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				dstptr[j] = resimg(i, j);
			}
			dstptr += 4 * getStride(ch);
		}
		ifedited = false;
	}



	if (ifcheck)
	{
		cv::Mat_<unsigned char> tmpimg0 = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch), CV_8UC1);
		Pel* dstptr = m_PicYHalfIntered;
		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				tmpimg0(i, j) = dstptr[j] / 64;
			}
			dstptr += 4 * getStride(ch);
		}

		cv::Mat_<unsigned char> tmpimg11 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);
		dstptr = m_apiPicBuf[ch];
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg11(i, j) = dstptr[j];
			}
			dstptr += getStride(ch);
		}

	}



	const Int stride = getStride(ch);
	int totaloffset = m_ctuOffsetInBuffer[ch == 0 ? 0 : 1][ctuRSAddr] + m_subCuOffsetInBuffer[ch == 0 ? 0 : 1][g_auiZscanToRaster[uiAbsZorderIdx]];
	int rows = totaloffset / stride;
	int cols = totaloffset % stride;

	return m_PicYHalfIntered + m_marginY * 4 * stride * 4 + m_marginX * 4 + rows * 4 * stride * 4 + cols * 4;
}

#if place_RDO
Pel* TComPicYuv::getInterplaceAddr(const Int ctuRSAddr, const Int uiAbsZorderIdx, int qp, bool ifcheck)
{
	const ComponentID ch = ComponentID(0);

	if (m_otherside_PicYIntered == NULL || ifplaceedited)
	{
		if (m_otherside_PicYIntered == NULL)
		{
			m_otherside_PicYIntered = (Pel*)xMalloc(Pel, getStride(ch) * 4 * getTotalHeight(ch) * 4);
		}

		string basetmodelstr = "testH22.caffemodel";

		int tbestqp = 22;
		int qps[4] = { 22, 27, 32, 37 };
		for (int i = 0; i < 4; i++)
		{
			if (abs(qp - tbestqp)>abs(qp - qps[i]))
			{
				tbestqp = qps[i];
			}
		}

		basetmodelstr[5] = char(tbestqp / 10 + 48);
		basetmodelstr[6] = char(tbestqp % 10 + 48);

		Pel* srcptr = m_apiPicBuf[ch];
		Pel* dstptr = m_otherside_PicYIntered;
		cv::Mat_<float> tmpimg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_32FC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg(i, j) = srcptr[j] / 255.0;
			}
			srcptr += getStride(ch);
		}

		vector<cv::Mat> tmpimgs;
		cv::Mat_<float> copytmp = cv::Mat(getTotalHeight(ch) - 1, getStride(ch) - 1, CV_32FC1);
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				copytmp(i, j) = tmpimg(i, j);
			}
		}
		tmpimgs.push_back(copytmp.clone());
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 1; j < getStride(ch); j++)
			{
				copytmp(i, j - 1) = tmpimg(i, j);
			}
		}
		tmpimgs.push_back(copytmp.clone());
		for (int i = 1; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				copytmp(i - 1, j) = tmpimg(i, j);
			}
		}
		tmpimgs.push_back(copytmp.clone());
		for (int i = 1; i < getTotalHeight(ch); i++)
		{
			for (int j = 1; j < getStride(ch); j++)
			{
				copytmp(i - 1, j - 1) = tmpimg(i, j);
			}
		}
		tmpimgs.push_back(copytmp.clone());

		//string model_file1 = "check_test_4x.prototxt";
		//string trained_file1 = "check_test_4x.caffemodel";
		//Interpolator interpolatort(model_file1, trained_file1, false);
		//interpolatort.interpolate_test();


		cv::Mat_<Short> resimg = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch), CV_16SC1);
		cv::Mat_<Short> tmpimg1 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_16SC1);
		std::vector<cv::Mat>  predictions;
#if place_RDO_includehalf
		string model_file = "mfi_x2_wider_net_otherside.prototxt";
		basetmodelstr = "mfi_x2_otherside.caffemodel";
#else
		string model_file = "fast_interp_deploy_2x_l20.prototxt";
#endif

		clock_t start_time = clock();
		Interpolator interpolator(model_file, basetmodelstr, false);
#if place_RDO_includehalf
		predictions = interpolator.interpolate_otherside(tmpimgs);
#else
		predictions = interpolator.interpolate(tmpimg);
#endif

		clock_t end_time = clock();
		printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

		tmpimg1 = predictions[0] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch)-1; i++)
		{
			for (int j = 0; j < getStride(ch)-1; j++)
			{
				resimg(i * 4, j * 4) = tmpimg(i, j) * 255 * 64;
				resimg(i * 4 + 2, j * 4) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[1] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch)-1; i++)
		{
			for (int j = 0; j < getStride(ch)-1; j++)
			{
				resimg(i * 4, j * 4 + 2) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[2] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch)-1; i++)
		{
			for (int j = 0; j < getStride(ch)-1; j++)
			{
				resimg(i * 4 + 2, j * 4 + 2) = tmpimg1(i, j);
			}
		}

		model_file = "mfi_x4_otherside_net.prototxt";
		basetmodelstr = "mfi_x4_otherside.caffemodel";
		

		start_time = clock();
		Interpolator interpolator_4x(model_file, basetmodelstr, false);
		predictions.clear();
		predictions = interpolator_4x.interpolate_4x_c_otherside(tmpimgs);
		end_time = clock();
		printf("The 4x running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

		tmpimg1.release();
		tmpimg1 = cv::Mat(getTotalHeight(ch)-1, getStride(ch)-1, CV_16SC1);
		tmpimg1 = predictions[0] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch)-1; i++)
		{
			for (int j = 0; j < getStride(ch)-1; j++)
			{
				resimg(i * 4, j * 4 + 1) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[1] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4, j * 4+3) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[2] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 1, j * 4 ) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[3] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 1, j * 4 + 1) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[4] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4+1, j * 4 + 2) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[5] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 1, j * 4 +3) = tmpimg1(i, j);
			}
		}

		tmpimg1 = predictions[6] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 2, j * 4+1) = tmpimg1(i, j);
			}
		}

		tmpimg1 = predictions[7] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 2, j * 4 + 3) = tmpimg1(i, j);
			}
		}

		tmpimg1 = predictions[8] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 +3, j * 4) = tmpimg1(i, j);
			}
		}

		tmpimg1 = predictions[9] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 3, j * 4 +1) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[10] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 + 3, j * 4 + 2) = tmpimg1(i, j);
			}
		}
		tmpimg1 = predictions[11] * 255 * 64;
		for (int i = 0; i < getTotalHeight(ch) - 1; i++)
		{
			for (int j = 0; j < getStride(ch) - 1; j++)
			{
				resimg(i * 4 +3, j * 4 +3) = tmpimg1(i, j);
			}
		}

		for (int i = 0; i < resimg.rows; i++)
		{
			for (int j = 0; j < resimg.cols; j++)
			{
				if (resimg(i, j) >= 255 * 64)
				{
					resimg(i, j) = 255 * 64;
				}
				if (resimg(i, j) < 0)
				{
					resimg(i, j) = 0;
				}
			}
		}

		cv::Mat tmp;
		resimg.convertTo(tmp, CV_16UC1);
		tmp = tmp / 64;

		cv::Mat tmpchar;
		tmp.convertTo(tmpchar, CV_8UC1);

		cv::Mat_<uchar> tmpchar_2x = cv::Mat(2 * getTotalHeight(ch), 2 * getStride(ch), CV_16SC1);
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpchar_2x(i * 2, j * 2) = resimg(i * 4, j * 4) / 64;
				tmpchar_2x(i * 2 + 1, j * 2) = resimg(i * 4 + 2, j * 4) / 64;
				tmpchar_2x(i * 2, j * 2 + 1) = resimg(i * 4, j * 4 + 2) / 64;
				tmpchar_2x(i * 2 + 1, j * 2 + 1) = resimg(i * 4 + 2, j * 4 + 2) / 64;
			}
		}


		predictions.clear();
		tmpimg.release();
		tmpimgs.clear();
		tmpimg1.release();

		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				dstptr[j] = resimg(i, j);
			}
			dstptr += 4 * getStride(ch);
		}
		ifplaceedited = false;
	}

	if (ifcheck)
	{
		cv::Mat_<unsigned char> tmpimg0 = cv::Mat(4 * getTotalHeight(ch), 4 * getStride(ch), CV_8UC1);
		Pel* dstptr = m_otherside_PicYIntered;
		for (int i = 0; i < 4 * getTotalHeight(ch); i++)
		{
			for (int j = 0; j < 4 * getStride(ch); j++)
			{
				tmpimg0(i, j) = dstptr[j] / 64;
			}
			dstptr += 4 * getStride(ch);
		}

		cv::Mat_<unsigned char> tmpimg11 = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);
		dstptr = m_apiPicBuf[ch];
		for (int i = 0; i < getTotalHeight(ch); i++)
		{
			for (int j = 0; j < getStride(ch); j++)
			{
				tmpimg11(i, j) = dstptr[j];
			}
			dstptr += getStride(ch);
		}

	}

	const Int stride = getStride(ch);
	int totaloffset = m_ctuOffsetInBuffer[ch == 0 ? 0 : 1][ctuRSAddr] + m_subCuOffsetInBuffer[ch == 0 ? 0 : 1][g_auiZscanToRaster[uiAbsZorderIdx]];
	int rows = totaloffset / stride;
	int cols = totaloffset % stride;

	return m_otherside_PicYIntered + m_marginY * 4 * stride * 4 + m_marginX * 4 + rows * 4 * stride * 4 + cols * 4;

}
#endif
#endif


Void TComPicYuv::createWithoutCUInfo(const Int picWidth,                 ///< picture width
	const Int picHeight,                ///< picture height
	const ChromaFormat chromaFormatIDC, ///< chroma format
	const Bool bUseMargin,              ///< if true, then a margin of uiMaxCUWidth+16 and uiMaxCUHeight+16 is created around the image.
	const UInt maxCUWidth,              ///< used for margin only
	const UInt maxCUHeight)             ///< used for margin only

{
	m_picWidth = picWidth;
	m_picHeight = picHeight;
	m_chromaFormatIDC = chromaFormatIDC;
	m_marginX = (bUseMargin ? maxCUWidth : 0) + 16;   // for 16-byte alignment
	m_marginY = (bUseMargin ? maxCUHeight : 0) + 16;  // margin for 8-tap filter and infinite padding
	m_bIsBorderExtended = false;

	// assign the picture arrays and set up the ptr to the top left of the original picture
	for (UInt comp = 0; comp < getNumberValidComponents(); comp++)
	{
		const ComponentID ch = ComponentID(comp);
		m_apiPicBuf[comp] = (Pel*)xMalloc(Pel, getStride(ch) * getTotalHeight(ch));
		m_piPicOrg[comp] = m_apiPicBuf[comp] + (m_marginY >> getComponentScaleY(ch)) * getStride(ch) + (m_marginX >> getComponentScaleX(ch));
	}
	// initialize pointers for unused components to NULL
	for (UInt comp = getNumberValidComponents(); comp < MAX_NUM_COMPONENT; comp++)
	{
		m_apiPicBuf[comp] = NULL;
		m_piPicOrg[comp] = NULL;
	}

	for (Int chan = 0; chan < MAX_NUM_CHANNEL_TYPE; chan++)
	{
		m_ctuOffsetInBuffer[chan] = NULL;
		m_subCuOffsetInBuffer[chan] = NULL;
	}
}



Void TComPicYuv::create(const Int picWidth,                 ///< picture width
	const Int picHeight,                ///< picture height
	const ChromaFormat chromaFormatIDC, ///< chroma format
	const UInt maxCUWidth,              ///< used for generating offsets to CUs.
	const UInt maxCUHeight,             ///< used for generating offsets to CUs.
	const UInt maxCUDepth,              ///< used for generating offsets to CUs.
	const Bool bUseMargin)              ///< if true, then a margin of uiMaxCUWidth+16 and uiMaxCUHeight+16 is created around the image.

{
	createWithoutCUInfo(picWidth, picHeight, chromaFormatIDC, bUseMargin, maxCUWidth, maxCUHeight);


	const Int numCuInWidth = m_picWidth / maxCUWidth + (m_picWidth  % maxCUWidth != 0);
	const Int numCuInHeight = m_picHeight / maxCUHeight + (m_picHeight % maxCUHeight != 0);
	for (Int chan = 0; chan < MAX_NUM_CHANNEL_TYPE; chan++)
	{
		const ChannelType ch = ChannelType(chan);
		const Int ctuHeight = maxCUHeight >> getChannelTypeScaleY(ch);
		const Int ctuWidth = maxCUWidth >> getChannelTypeScaleX(ch);
		const Int stride = getStride(ch);

		m_ctuOffsetInBuffer[chan] = new Int[numCuInWidth * numCuInHeight];

		for (Int cuRow = 0; cuRow < numCuInHeight; cuRow++)
		{
			for (Int cuCol = 0; cuCol < numCuInWidth; cuCol++)
			{
				m_ctuOffsetInBuffer[chan][cuRow * numCuInWidth + cuCol] = stride * cuRow * ctuHeight + cuCol * ctuWidth;
			}
		}

		m_subCuOffsetInBuffer[chan] = new Int[(size_t)1 << (2 * maxCUDepth)];

		const Int numSubBlockPartitions = (1 << maxCUDepth);
		const Int minSubBlockHeight = (ctuHeight >> maxCUDepth);
		const Int minSubBlockWidth = (ctuWidth >> maxCUDepth);

		for (Int buRow = 0; buRow < numSubBlockPartitions; buRow++)
		{
			for (Int buCol = 0; buCol < numSubBlockPartitions; buCol++)
			{
				m_subCuOffsetInBuffer[chan][(buRow << maxCUDepth) + buCol] = stride  * buRow * minSubBlockHeight + buCol * minSubBlockWidth;
			}
		}
	}
}

Void TComPicYuv::destroy()
{
	for (Int comp = 0; comp < MAX_NUM_COMPONENT; comp++)
	{
		m_piPicOrg[comp] = NULL;

		if (m_apiPicBuf[comp])
		{
			xFree(m_apiPicBuf[comp]);
			m_apiPicBuf[comp] = NULL;
		}
#if enable_deep_interpolation
		if (comp == 0 && (m_PicYHalfIntered!=NULL)){ xFree(m_PicYHalfIntered); m_PicYHalfIntered = NULL; }
#if place_RDO
		if (comp == 0 && (m_otherside_PicYIntered != NULL)){ xFree(m_otherside_PicYIntered); m_otherside_PicYIntered = NULL; }
#endif
#endif
	}

	for (UInt chan = 0; chan < MAX_NUM_CHANNEL_TYPE; chan++)
	{
		if (m_ctuOffsetInBuffer[chan])
		{
			delete[] m_ctuOffsetInBuffer[chan];
			m_ctuOffsetInBuffer[chan] = NULL;
		}
		if (m_subCuOffsetInBuffer[chan])
		{
			delete[] m_subCuOffsetInBuffer[chan];
			m_subCuOffsetInBuffer[chan] = NULL;
		}
	}
}



Void  TComPicYuv::copyToPic(TComPicYuv*  pcPicYuvDst) const
{
	assert(m_chromaFormatIDC == pcPicYuvDst->getChromaFormat());

	for (Int comp = 0; comp < getNumberValidComponents(); comp++)
	{
		const ComponentID compId = ComponentID(comp);
		const Int width = getWidth(compId);
		const Int height = getHeight(compId);
		const Int strideSrc = getStride(compId);
		assert(pcPicYuvDst->getWidth(compId) == width);
		assert(pcPicYuvDst->getHeight(compId) == height);
		if (strideSrc == pcPicYuvDst->getStride(compId))
		{
			::memcpy(pcPicYuvDst->getBuf(compId), getBuf(compId), sizeof(Pel)*strideSrc*getTotalHeight(compId));
		}
		else
		{
			const Pel *pSrc = getAddr(compId);
			Pel *pDest = pcPicYuvDst->getAddr(compId);
			const UInt strideDest = pcPicYuvDst->getStride(compId);

			for (Int y = 0; y < height; y++, pSrc += strideSrc, pDest += strideDest)
			{
				::memcpy(pDest, pSrc, width*sizeof(Pel));
			}
		}
	}
#if enable_deep_interpolation
	pcPicYuvDst->ifedited = true;
#if place_RDO
	pcPicYuvDst->ifplaceedited = true;
#endif
#endif
}


Void TComPicYuv::extendPicBorder()
{
	if (m_bIsBorderExtended)
	{
		return;
	}

	for (Int comp = 0; comp < getNumberValidComponents(); comp++)
	{
		const ComponentID compId = ComponentID(comp);
		Pel *piTxt = getAddr(compId); // piTxt = point to (0,0) of image within bigger picture.
		const Int stride = getStride(compId);
		const Int width = getWidth(compId);
		const Int height = getHeight(compId);
		const Int marginX = getMarginX(compId);
		const Int marginY = getMarginY(compId);

		Pel*  pi = piTxt;
		// do left and right margins
		for (Int y = 0; y < height; y++)
		{
			for (Int x = 0; x < marginX; x++)
			{
				pi[-marginX + x] = pi[0];
				pi[width + x] = pi[width - 1];
			}
			pi += stride;
		}

		// pi is now the (0,height) (bottom left of image within bigger picture
		pi -= (stride + marginX);
		// pi is now the (-marginX, height-1)
		for (Int y = 0; y < marginY; y++)
		{
			::memcpy(pi + (y + 1)*stride, pi, sizeof(Pel)*(width + (marginX << 1)));
		}

		// pi is still (-marginX, height-1)
		pi -= ((height - 1) * stride);
		// pi is now (-marginX, 0)
		for (Int y = 0; y < marginY; y++)
		{
			::memcpy(pi - (y + 1)*stride, pi, sizeof(Pel)*(width + (marginX << 1)));
		}
	}

	m_bIsBorderExtended = true;
}



// NOTE: This function is never called, but may be useful for developers.
Void TComPicYuv::dump(const std::string &fileName, const BitDepths &bitDepths, const Bool bAppend, const Bool bForceTo8Bit) const
{
	FILE *pFile = fopen(fileName.c_str(), bAppend ? "ab" : "wb");

	Bool is16bit = false;
	for (Int comp = 0; comp < getNumberValidComponents() && !bForceTo8Bit; comp++)
	{
		if (bitDepths.recon[toChannelType(ComponentID(comp))]>8)
		{
			is16bit = true;
		}
	}

	for (Int comp = 0; comp < getNumberValidComponents(); comp++)
	{
		const ComponentID  compId = ComponentID(comp);
		const Pel         *pi = getAddr(compId);
		const Int          stride = getStride(compId);
		const Int          height = getHeight(compId);
		const Int          width = getWidth(compId);

		if (is16bit)
		{
			for (Int y = 0; y < height; y++)
			{
				for (Int x = 0; x < width; x++)
				{
					UChar uc = (UChar)((pi[x] >> 0) & 0xff);
					fwrite(&uc, sizeof(UChar), 1, pFile);
					uc = (UChar)((pi[x] >> 8) & 0xff);
					fwrite(&uc, sizeof(UChar), 1, pFile);
				}
				pi += stride;
			}
		}
		else
		{
			const Int shift = bitDepths.recon[toChannelType(compId)] - 8;
			const Int offset = (shift > 0) ? (1 << (shift - 1)) : 0;
			for (Int y = 0; y < height; y++)
			{
				for (Int x = 0; x < width; x++)
				{
					UChar uc = (UChar)Clip3<Pel>(0, 255, (pi[x] + offset) >> shift);
					fwrite(&uc, sizeof(UChar), 1, pFile);
				}
				pi += stride;
			}
		}
	}

	fclose(pFile);
}

//! \}
